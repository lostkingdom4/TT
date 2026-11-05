"""Exhaustive grid-search autotuning for one (operator, kernel).

For every shape produced by the TritonBench operator this runner:

1. Measures the **original** (no-injector) latency as the comparison
   baseline.
2. Builds the config list:
   * autotuned kernels → captured from the kernel's own
     ``@triton.autotune`` decorator on a warm-up launch
   * non-autotuned kernels → ``itertools.product`` of
     ``get_search_space(operator, kernel)``
3. For each config, installs it via ``_FixedConfigInjector`` and
   benchmarks with ``triton.testing.do_bench(warmup=5, rep=20,
   return_mode="all")`` after an L2 cache flush — identical to
   ``compare_configs._benchmark_config`` in ``batch_runner``.
4. Flushes the JSON envelope to ``--output`` after **every trial** so
   a ``Ctrl-C`` never loses more than one trial's worth of work.

On resume, already-recorded trials (by ``(shape_id, trial_id)``) are
skipped.
"""

import argparse
import copy
import gc
import itertools
import json
import logging
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.common.injector import _FixedConfigInjector
from baselines.common.search_space import (
    CONFIG_LEVEL_PARAMS,
    get_autotune_configs,
    get_search_space,
)
from baselines.common.tritonbench_setup import (
    build_tb_args,
    fresh_cache_context,
    jsonable,
    load_op_class,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# benchmark primitives (mirror baselines/bo/operator_runner.py, which in
# turn mirrors compare_configs._benchmark_config / _flush_l2_cache)
# ---------------------------------------------------------------------------

def _cast_floats_to_fp32(val):
    """Recursively cast every floating-point tensor leaf to torch.float32.

    Used by the tritonbench-matches-full experiment: TritonBench default
    is fp16, but compare_configs (Pipeline A) feeds kernels the dtype
    recorded in its shapes_file (fp32). To neutralize that axis when
    comparing, we force TritonBench's example_inputs to fp32 as well
    before each shape's bench trials. Non-float leaves are returned as-is.
    """
    import torch
    if isinstance(val, torch.Tensor):
        if val.is_floating_point():
            return val.to(torch.float32)
        return val
    if isinstance(val, dict):
        return {k: _cast_floats_to_fp32(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        casted = [_cast_floats_to_fp32(v) for v in val]
        return type(val)(casted) if not isinstance(val, list) else casted
    return val


def _compute_stats(times: list) -> Dict[str, float]:
    if not times:
        return {}
    s = sorted(times)
    n = len(s)
    return {
        "median": float(s[n // 2]),
        "mean": float(statistics.mean(s)),
        "min": float(s[0]),
        "max": float(s[-1]),
        "std_dev": float(statistics.stdev(s)) if n > 1 else 0.0,
        "samples": n,
    }


def _flush_l2_cache(device: str = "cuda") -> None:
    import torch
    flush_size = 48 * 1024 * 1024 // 4  # 48 MB of float32
    buf = torch.empty(flush_size, dtype=torch.float32, device=device)
    buf.zero_()
    torch.cuda.synchronize()


def _bench_once(
    impl_method,
    example_inputs,
    warmup: int,
    rep: int,
) -> Optional[Dict[str, float]]:
    import torch
    import triton

    try:
        if isinstance(example_inputs, dict):
            fn = impl_method(**example_inputs)
        else:
            fn = impl_method(*example_inputs)
        if not callable(fn):
            return None
        # Prime any runtime autotuner cache with one untimed launch so a
        # cold `@triton.autotune` sweep (which can take hundreds of ms
        # for kernels with many configs, e.g. jag_softmax at ~600 ms)
        # doesn't bleed into the do_bench measurement window. For
        # non-autotuned kernels this is just a harmless extra launch.
        try:
            fn()
            torch.cuda.synchronize()
        except Exception:
            pass
        _flush_l2_cache()
        times = triton.testing.do_bench(
            fn, warmup=warmup, rep=rep, return_mode="all",
        )
    except Exception as exc:
        log.debug("do_bench failed: %s", exc)
        return None

    if times is None:
        return None
    if hasattr(times, "tolist"):
        times = times.tolist()
    if not times:
        return None
    return _compute_stats(times)


def _find_triton_impl(op) -> Optional[str]:
    try:
        from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS
        op_name = getattr(op, "name", None) or type(op).__name__.lower()
        benchmarks = REGISTERED_BENCHMARKS.get(op_name, {})
        for name in benchmarks:
            if "triton" in name.lower():
                return name
        if benchmarks:
            return next(iter(benchmarks))
    except Exception as exc:
        log.debug("Could not read REGISTERED_BENCHMARKS: %s", exc)
    return None


# ---------------------------------------------------------------------------
# config-list construction
# ---------------------------------------------------------------------------

def _build_config_list(
    operator: str,
    kernel: Optional[str],
    has_autotune: bool,
    force_full_space: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Return ``(configs, source)`` for the given kernel.

    * ``force_full_space=True`` → always use the Cartesian product of
      ``KERNEL_SEARCH_SPACES`` (the "full search") even if the kernel is
      in ``KERNEL_AUTOTUNE_CONFIGS``. Used by the separate full-grid
      baseline that sweeps every launch-config combination on GEMM-like
      kernels.
    * autotune-wrapped kernels (``has_autotune=True`` and listed in
      ``KERNEL_AUTOTUNE_CONFIGS``) → verbatim list from the decorator.
    * else → Cartesian product of ``get_search_space(operator, kernel)``.
    """
    if has_autotune and not force_full_space:
        explicit = get_autotune_configs(operator, kernel)
        if explicit is not None:
            return [dict(c) for c in explicit], "autotune_decorator"
    space = get_search_space(operator, kernel)
    source = "common_search_space_full" if force_full_space else "common_search_space"
    return _enumerate_search_space(space), source


def _enumerate_search_space(space: Dict[str, List]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


# ---------------------------------------------------------------------------
# partial-result loading for resume
# ---------------------------------------------------------------------------

def _load_partial(output_path: Path) -> Tuple[List[Dict[str, Any]], Set[Tuple[int, int]]]:
    """Read an existing envelope and return (shape_results, done_trials).

    ``done_trials`` is the set of ``(input_id, trial_id)`` pairs already
    benchmarked — we skip those on resume.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return [], set()
    try:
        with open(output_path) as f:
            data = json.load(f)
    except Exception:
        return [], set()

    shapes = data.get("shapes", [])
    done: Set[Tuple[int, int]] = set()
    for s in shapes:
        sid = s.get("input_id")
        if sid is None:
            continue
        for t in s.get("trial_history", []):
            tid = t.get("trial_id")
            if tid is not None:
                done.add((int(sid), int(tid)))
    return shapes, done


# ---------------------------------------------------------------------------
# envelope flusher
# ---------------------------------------------------------------------------

def _flush_envelope(
    output_path: Path,
    operator: str,
    kernel: Optional[str],
    shape_results: List[Dict[str, Any]],
    has_autotune: bool,
    status: str,
) -> None:
    payload = {
        "operator": operator,
        "kernel": kernel,
        "num_shapes": len(shape_results),
        "status": status,
        "has_autotune": has_autotune,
        "shapes": shape_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, output_path)


# ---------------------------------------------------------------------------
# shape-description helper
# ---------------------------------------------------------------------------

def _describe_shape(example_inputs) -> str:
    try:
        if isinstance(example_inputs, dict):
            import torch
            parts = []
            for k, v in example_inputs.items():
                if isinstance(v, torch.Tensor):
                    parts.append(f"{k}={list(v.shape)}")
                elif isinstance(v, (int, float, bool)):
                    parts.append(f"{k}={v}")
            return ", ".join(parts) if parts else str(example_inputs)
        return str(example_inputs)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# core runner
# ---------------------------------------------------------------------------

def run_operator(
    operator: str,
    kernel: Optional[str],
    output_path: Path,
    has_autotune: bool = False,
    device: str = "cuda",
    warmup: int = 5,
    rep: int = 20,
    force_full_space: bool = False,
) -> List[Dict[str, Any]]:
    """Grid-search every shape of ``(operator, kernel)``; persist per trial."""
    import torch

    tb_args, extra_args = build_tb_args(
        device=device, precision="fp16",
        metrics="latency",
        warmup=warmup, rep=rep,
    )
    OpClass = load_op_class(operator)

    existing_shapes, done_trials = _load_partial(output_path)
    existing_by_id: Dict[int, Dict[str, Any]] = {}
    for s in existing_shapes:
        sid = s.get("input_id")
        if sid is not None:
            existing_by_id[int(sid)] = s

    results: List[Dict[str, Any]] = list(existing_shapes)

    def _flush(status: str = "in_progress"):
        _flush_envelope(
            output_path, operator, kernel, results,
            has_autotune=has_autotune, status=status,
        )

    # ``has_autotune`` comes from discover (operator-level detection) and
    # may be True for a sibling kernel of an @triton.autotune-wrapped one.
    # We treat a kernel as autotune-wrapped only if we have an explicit
    # KERNEL_AUTOTUNE_CONFIGS entry for it; otherwise we fall back to the
    # common search space and JIT-path injection.
    effective_has_autotune = (
        has_autotune and get_autotune_configs(operator, kernel) is not None
    )

    with fresh_cache_context():
        try:
            op = OpClass(tb_args=tb_args, extra_args=extra_args)
        except Exception as exc:
            log.error("Failed to instantiate %s: %s", operator, exc)
            results.append(_error_shape(operator, kernel, -1, str(exc)))
            _flush("failed")
            return results

        triton_impl = _find_triton_impl(op)
        if triton_impl is None:
            results.append(_error_shape(
                operator, kernel, -1,
                "No Triton implementation registered",
            ))
            _flush("failed")
            return results

        impl_method = getattr(op, triton_impl, None)
        if impl_method is None:
            results.append(_error_shape(
                operator, kernel, -1,
                f"Operator has no method {triton_impl}",
            ))
            _flush("failed")
            return results

        # --- Build the config list ---
        config_list, config_source = _build_config_list(
            operator, kernel, effective_has_autotune,
            force_full_space=force_full_space,
        )
        n_configs = len(config_list)
        log.info("Grid-searching %d configs per shape (source=%s, "
                 "has_autotune=%s -> effective=%s)",
                 n_configs, config_source, has_autotune,
                 effective_has_autotune)

        # The first config is the reference config for t_orig. We use
        # the SAME _FixedConfigInjector + do_bench methodology as the
        # trial measurements, so t_orig is directly comparable (same
        # measurement convention as batch_runner / compare_configs.py).
        if n_configs == 0:
            results.append(_error_shape(
                operator, kernel, -1, "empty config list",
            ))
            _flush("failed")
            return results
        reference_config = config_list[0]

        shape_idx = 0
        while True:
            try:
                example_inputs = op.get_example_inputs()
            except Exception as exc:
                log.error("get_example_inputs() failed at shape %d: %s",
                          shape_idx, exc)
                break
            if example_inputs is None:
                break

            # Cast every float tensor in example_inputs to fp32 so that the
            # aligned re-measurement (and every trial via _FixedConfigInjector)
            # runs on fp32 tensors — matching compare_configs Pipeline A's
            # shapes_file-derived fp32 inputs. Pipeline A is the reference
            # point for this branch; dropping tensor width back to fp16 here
            # would defeat the experiment.
            example_inputs = _cast_floats_to_fp32(example_inputs)

            x_val = _describe_shape(example_inputs)

            shape_rec = existing_by_id.get(shape_idx)
            if shape_rec is None:
                shape_rec = {
                    "operator": operator,
                    "kernel": kernel,
                    "input_id": shape_idx,
                    "x_val": x_val,
                    "has_autotune": has_autotune,
                    "triton_impl_name": triton_impl,
                    "n_configs": n_configs,
                    "config_source": config_source,
                    "reference_config": None,
                    "original_config_latency_ms": None,
                    "original_stats": None,
                    "best_config": None,
                    "best_latency_ms": None,
                    "latency_ms": None,
                    "best_trial_id": None,
                    "speedup": None,
                    "trial_history": [],
                    "trial_errors": 0,
                    "status": "in_progress",
                    "error": None,
                    "verified": False,
                    "verification_error": None,
                }
                results.append(shape_rec)
                existing_by_id[shape_idx] = shape_rec
            else:
                shape_rec["n_configs"] = n_configs
                shape_rec["config_source"] = config_source

            # --- 1. Reference baseline (= first config in the list) ---
            # Measured under _FixedConfigInjector + do_bench, same as
            # every trial below, so t_orig is directly comparable.
            # NOT the same as Triton's native-autotuner pick (which is
            # noisy on V100 for GEMM shapes).
            if shape_rec.get("original_config_latency_ms") is None:
                try:
                    with _FixedConfigInjector(
                        operator, reference_config,
                        effective_has_autotune, kernel=kernel,
                    ):
                        orig = _bench_once(impl_method, example_inputs,
                                           warmup=warmup, rep=rep)
                except Exception as exc:
                    log.debug("Reference-config bench failed: %s", exc)
                    orig = None
                if orig is not None:
                    shape_rec["original_stats"] = orig
                    shape_rec["original_config_latency_ms"] = orig.get("median")
                shape_rec["reference_config"] = copy.deepcopy(reference_config)
                _flush()

            # --- 2. Grid-search every config ---
            for trial_id, cfg in enumerate(config_list):
                if (shape_idx, trial_id) in done_trials:
                    continue

                trial_errors_before = shape_rec["trial_errors"]
                try:
                    with _FixedConfigInjector(
                        operator, cfg, effective_has_autotune, kernel=kernel,
                    ):
                        stats = _bench_once(impl_method, example_inputs,
                                            warmup=warmup, rep=rep)
                    if stats is None:
                        raise RuntimeError("do_bench returned None")
                    lat = stats.get("median")
                    trial_status = "success" if lat is not None else "error"
                    trial_error = None
                except Exception as exc:
                    lat = None
                    trial_status = "error"
                    trial_error = str(exc)
                    shape_rec["trial_errors"] = trial_errors_before + 1

                shape_rec["trial_history"].append({
                    "trial_id": trial_id,
                    "config": copy.deepcopy(cfg),
                    "latency_ms": lat,
                    "status": trial_status,
                    "error": trial_error,
                })
                done_trials.add((shape_idx, trial_id))
                _flush()   # per-trial flush for resume safety

            # --- 3. Pick best config from trial history ---
            best_lat = None
            best_cfg = None
            best_id = None
            for t in shape_rec["trial_history"]:
                lat = t.get("latency_ms")
                if lat is None:
                    continue
                if best_lat is None or lat < best_lat:
                    best_lat = lat
                    best_cfg = t.get("config")
                    best_id = t.get("trial_id")

            shape_rec["best_config"] = best_cfg
            shape_rec["best_latency_ms"] = best_lat
            shape_rec["latency_ms"] = best_lat
            shape_rec["best_trial_id"] = best_id

            orig_lat = shape_rec.get("original_config_latency_ms")
            if orig_lat is not None and best_lat is not None and best_lat > 0:
                shape_rec["speedup"] = orig_lat / best_lat

            if best_cfg is not None:
                shape_rec["status"] = "success"
                shape_rec["verified"] = True
                shape_rec["error"] = None
            else:
                shape_rec["status"] = "error"
                shape_rec["verified"] = False
                shape_rec["error"] = f"All {n_configs} trials failed"

            _flush()
            shape_idx += 1

        any_ok = any(r.get("status") == "success" for r in results)
        _flush("completed" if any_ok else "failed")

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    return results


def _error_shape(
    operator: str, kernel: Optional[str], input_id: int, error: str,
) -> Dict[str, Any]:
    return {
        "operator": operator,
        "kernel": kernel,
        "input_id": input_id,
        "x_val": str(input_id),
        "has_autotune": False,
        "triton_impl_name": None,
        "n_configs": 0,
        "config_source": None,
        "original_config_latency_ms": None,
        "original_stats": None,
        "best_config": None,
        "best_latency_ms": None,
        "latency_ms": None,
        "best_trial_id": None,
        "speedup": None,
        "trial_history": [],
        "trial_errors": 0,
        "status": "error",
        "error": error,
        "verified": False,
        "verification_error": error,
    }


# ---------------------------------------------------------------------------
# subprocess entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive grid-search autotuner for one (operator, kernel)",
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--kernel", default=None,
                        help="Kernel name — selects per-kernel search space")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5,
                        help="do_bench warmup (default: 5, matches batch_runner)")
    parser.add_argument("--rep", type=int, default=20,
                        help="do_bench rep (default: 20, matches batch_runner)")
    parser.add_argument("--has-autotune", action="store_true", default=False)
    parser.add_argument("--full-space", action="store_true", default=False,
                        help="Force the full Cartesian KERNEL_SEARCH_SPACES "
                             "even for @triton.autotune-wrapped kernels. Used "
                             "by the separate full-grid baseline.")
    parser.add_argument("--output", required=True,
                        help="Final JSON path — written atomically after "
                             "every trial so runs are resume-safe.")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.time()
    output_path = Path(args.output).resolve()
    try:
        results = run_operator(
            operator=args.operator,
            kernel=args.kernel,
            output_path=output_path,
            has_autotune=args.has_autotune,
            device=args.device,
            warmup=args.warmup,
            rep=args.rep,
            force_full_space=args.full_space,
        )
    except Exception as exc:
        log.error("Fatal error for %s/%s: %s", args.operator, args.kernel, exc)
        traceback.print_exc()
        results = [_error_shape(args.operator, args.kernel, -1, f"Fatal: {exc}")]
        _flush_envelope(
            output_path, args.operator, args.kernel, results,
            has_autotune=args.has_autotune, status="failed",
        )
    elapsed = time.time() - t0

    ok = sum(1 for r in results if r.get("status") == "success")
    err = sum(1 for r in results if r.get("status") == "error")
    improved = sum(
        1 for r in results
        if r.get("speedup") is not None and r["speedup"] > 1.005
    )
    print(
        f"[{args.operator}/{args.kernel}] shapes={len(results)}, "
        f"ok={ok}, errors={err}, improved={improved}, "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
