"""Per-(kernel, shape) Optuna TPE baseline runner.

For one (operator, kernel) pair this subprocess:

1. Loads the TritonBench operator.
2. For each shape it produces, measures the **original** latency
   (default config, no injector).
3. Runs an Optuna TPE study of ``n_trials`` iterations where each
   trial installs a sampled config via ``_FixedConfigInjector`` and
   benchmarks the shape with ``triton.testing.do_bench`` using the
   paper's ``warmup=5, rep=20, return_mode="all"`` settings.
4. Records the best config + speedup and emits a per-shape result
   dict with the full trial history.

Designed to be invoked from ``baselines.bo.run_all`` — one subprocess
per (operator, kernel) pair for GPU isolation.
"""

import argparse
import gc
import json
import logging
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.bo.bo_search import BOSearch
from baselines.common.budget import scaled_budget, space_size
from baselines.common.injector import _FixedConfigInjector
from baselines.common.search_space import get_search_space
from baselines.common.tritonbench_setup import (
    build_tb_args,
    fresh_cache_context,
    jsonable,
    load_op_class,
)

log = logging.getLogger(__name__)


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
    """Benchmark the kernel implementation once with do_bench.

    Returns the latency stats dict, or ``None`` on failure.
    """
    import triton

    try:
        if isinstance(example_inputs, dict):
            fn = impl_method(**example_inputs)
        else:
            fn = impl_method(*example_inputs)
        if not callable(fn):
            return None
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
    """Return the first registered Triton impl name on ``op``."""
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


def run_operator(
    operator: str,
    kernel: Optional[str],
    n_trials: int = 9,
    seed: int = 42,
    device: str = "cuda",
    warmup: int = 5,
    rep: int = 20,
    has_autotune: bool = False,
    budget_policy: str = "fixed",
) -> List[Dict[str, Any]]:
    """Run BO for every shape of ``(operator, kernel)`` and return results.

    ``budget_policy`` controls the per-shape trial count:

    * ``"fixed"``  — use ``n_trials`` directly (legacy behaviour, matches
      the LLM multi-agent's 9-eval budget).
    * ``"scaled"`` — compute per-kernel budget from the search-space size
      via ``scaled_budget`` (clip(round(0.10 * |space|), 9, 150)).
    """
    import torch

    tb_args, extra_args = build_tb_args(
        device=device, precision="fp16",
        metrics="latency",
        warmup=warmup, rep=rep,
    )
    OpClass = load_op_class(operator)

    results: List[Dict[str, Any]] = []
    search_space = get_search_space(operator, kernel)
    if budget_policy == "scaled":
        effective_trials = scaled_budget(space_size(search_space))
    else:
        effective_trials = int(n_trials)
    log.info(
        "Search space for %s/%s: %s | budget_policy=%s | trials=%d",
        operator, kernel, search_space, budget_policy, effective_trials,
    )

    with fresh_cache_context():
        try:
            op = OpClass(tb_args=tb_args, extra_args=extra_args)
        except Exception as exc:
            log.error("Failed to instantiate %s: %s", operator, exc)
            return [_error_result(operator, kernel, -1, effective_trials, str(exc))]

        triton_impl = _find_triton_impl(op)
        if triton_impl is None:
            return [_error_result(
                operator, kernel, -1, effective_trials,
                "No Triton implementation registered",
            )]

        impl_method = getattr(op, triton_impl, None)
        if impl_method is None:
            return [_error_result(
                operator, kernel, -1, effective_trials,
                f"Operator has no method {triton_impl}",
            )]

        log.info("Running BO for %s / %s (impl=%s, trials=%d, policy=%s)",
                 operator, kernel, triton_impl, effective_trials, budget_policy)

        shape_idx = 0
        while True:
            example_inputs = op.get_example_inputs()
            if example_inputs is None:
                break

            # Best-effort x_val: the first scalar/tuple of the inputs dict.
            x_val = _describe_shape(example_inputs)

            # ---- 1. Baseline latency with original config ----
            orig_stats = _bench_once(impl_method, example_inputs,
                                     warmup=warmup, rep=rep)

            # ---- 2. BO loop ----
            search = BOSearch(
                search_space, n_trials=effective_trials, seed=seed,
            )
            per_trial_errors = 0
            for _ in range(effective_trials):
                cfg = search.propose()
                try:
                    with _FixedConfigInjector(operator, cfg, has_autotune,
                                              kernel=kernel):
                        stats = _bench_once(impl_method, example_inputs,
                                            warmup=warmup, rep=rep)
                    lat = stats.get("median") if stats else None
                except Exception as exc:
                    log.debug("Trial failed for cfg=%s: %s", cfg, exc)
                    lat = None
                    per_trial_errors += 1
                search.observe(lat)

            best_cfg, best_lat, best_id = search.best()
            orig_lat = orig_stats.get("median") if orig_stats else None
            speedup = (
                (orig_lat / best_lat)
                if (orig_lat is not None and best_lat is not None and best_lat > 0)
                else None
            )

            if best_cfg is None:
                status = "error"
                error_msg = f"All {effective_trials} BO trials failed"
            else:
                status = "success"
                error_msg = None

            results.append({
                "operator": operator,
                "kernel": kernel,
                "input_id": shape_idx,
                "x_val": x_val,
                "has_autotune": has_autotune,
                "triton_impl_name": triton_impl,
                "n_trials": effective_trials,
                "budget": effective_trials,
                "budget_policy": budget_policy,
                "n_startup_trials": search.n_startup_trials,
                "space_size": space_size(search_space),
                "bo_config": (
                    {str(k): jsonable(v) for k, v in best_cfg.items()}
                    if best_cfg else None
                ),
                "bo_best_trial_id": best_id,
                "bo_trial_history": search.history(),
                "bo_trial_errors": per_trial_errors,
                "original_config_latency_ms": orig_lat,
                "original_stats": orig_stats,
                "best_latency_ms": best_lat,
                "latency_ms": best_lat,
                "speedup": speedup,
                "status": status,
                "error": error_msg,
                # No cross-impl verification here — BO changes only the
                # launch config, not the algorithm.  Mark as verified
                # iff the trial produced a valid latency.
                "verified": status == "success",
                "verification_error": None,
            })

            shape_idx += 1

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    return results


def _describe_shape(example_inputs) -> str:
    """Human-readable shape tag for one TritonBench example-inputs object."""
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


def _error_result(
    operator: str, kernel: Optional[str], input_id: int,
    n_trials: int, error: str,
) -> Dict[str, Any]:
    return {
        "operator": operator,
        "kernel": kernel,
        "input_id": input_id,
        "x_val": str(input_id),
        "has_autotune": False,
        "triton_impl_name": None,
        "n_trials": n_trials,
        "bo_config": None,
        "bo_best_trial_id": None,
        "bo_trial_history": [],
        "bo_trial_errors": n_trials,
        "original_config_latency_ms": None,
        "original_stats": None,
        "best_latency_ms": None,
        "latency_ms": None,
        "speedup": None,
        "status": "error",
        "error": error,
        "verified": False,
        "verification_error": error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna TPE BO on one TritonBench (operator, kernel)",
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--kernel", default=None,
                        help="Kernel name — selects the per-kernel search space")
    parser.add_argument("--n-trials", type=int, default=9,
                        help="Optuna trials per shape (default: 9)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5,
                        help="do_bench warmup (default: 5, matches paper)")
    parser.add_argument("--rep", type=int, default=20,
                        help="do_bench rep (default: 20, matches paper)")
    parser.add_argument("--has-autotune", action="store_true", default=False)
    parser.add_argument("--budget-policy", default="fixed",
                        choices=["fixed", "scaled"],
                        help="fixed: use --n-trials; scaled: per-kernel budget "
                             "= clip(round(0.10 * |search space|), 9, 150).")
    parser.add_argument("--harness", default="compare_configs",
                        choices=["tritonbench", "compare_configs"],
                        help="compare_configs (default): match the paper's "
                             "Full-system measurement path using extracted "
                             "kernel .py + per-input shapes.json. "
                             "tritonbench: legacy TritonBench op.get_example_inputs "
                             "path (different input distribution).")
    parser.add_argument("--skip-shape-ids", type=str, default="",
                        help="Comma-separated input_ids to skip (used for retry after CUDA context poisoning)")
    parser.add_argument("--ir-dir", type=Path, default=None,
                        help="IR output directory (extract/ir_output/<device>); "
                             "required when --harness=compare_configs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-shape-timeout-s", type=float, default=None,
                        help="Wall-clock cap per shape (seconds). Remaining "
                             "trials after this budget expire are recorded "
                             "as failed. Default: max(300, 4*budget).")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.time()
    try:
        if args.harness == "compare_configs":
            if args.ir_dir is None:
                raise SystemExit("--ir-dir is required when --harness=compare_configs")
            from baselines.common.cc_bench import run_operator_cc
            results = run_operator_cc(
                operator=args.operator,
                kernel=args.kernel,
                strategy_name="bo",
                ir_dir=args.ir_dir,
                skip_shape_ids=set(int(x) for x in args.skip_shape_ids.split(",") if x.strip()),
                budget_policy=args.budget_policy,
                fixed_n_trials=args.n_trials,
                seed=args.seed,
                warmup=args.warmup,
                rep=args.rep,
                has_autotune=args.has_autotune,
                device=args.device,
                incremental_output=Path(args.output),
                per_shape_timeout_s=args.per_shape_timeout_s,
            )
        else:
            results = run_operator(
                operator=args.operator,
                kernel=args.kernel,
                n_trials=args.n_trials,
                seed=args.seed,
                device=args.device,
                warmup=args.warmup,
                rep=args.rep,
                has_autotune=args.has_autotune,
                budget_policy=args.budget_policy,
            )
    except Exception as exc:
        log.error("Fatal error for %s/%s: %s", args.operator, args.kernel, exc)
        traceback.print_exc()
        results = [_error_result(args.operator, args.kernel, -1,
                                 args.n_trials, f"Fatal: {exc}")]
    elapsed = time.time() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    ok = sum(1 for r in results if r.get("status") == "success")
    err = sum(1 for r in results if r.get("status") == "error")
    with_speedup = sum(
        1 for r in results
        if r.get("speedup") is not None and r["speedup"] > 1.005
    )
    print(
        f"[{args.operator}/{args.kernel}] shapes={len(results)}, "
        f"ok={ok}, errors={err}, improved={with_speedup}, "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
