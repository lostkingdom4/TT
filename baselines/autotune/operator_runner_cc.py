"""Autotune grid-search baseline using compare_configs harness.

Same public schema as ``operator_runner.py`` but replaces the TritonBench
operator-wrapper benchmark path with a direct
``compare_configs.benchmark_config_pair`` call. That matches the exact
measurement path the multi-agent Full pipeline uses (per-shape
shapes_file + per-input kernel .py + do_bench w/ L2 flush), so the
``best_latency_ms`` reported here is apples-to-apples with the paper's
Full-system numbers.

The *reference* (t_orig) per shape is the **baked-in** config from
``extract/ir_output/<device>/<op>/<op>_<kernel>_input<N>_shapes.json``,
which is also what Full uses for its ``original_median_ms``.

Fields written into per-kernel JSONs are unchanged, so downstream
analysis scripts keep working:

    reference_config          — baked-in kwargs (dict)
    original_config_latency_ms
    original_stats
    best_config / best_latency_ms / speedup / trial_history / ...
"""

import argparse
import copy
import gc
import glob
import itertools
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.autotune.operator_runner import (  # noqa: E402
    _flush_envelope, _load_partial, _error_shape,
)
from baselines.common.search_space import (  # noqa: E402
    get_autotune_configs, get_search_space,
)

log = logging.getLogger(__name__)

_META_KEYS = {"num_warps", "num_stages", "num_ctas"}
_EXCLUDE_FROM_BAKED_IN = {
    "warmup", "ALLOW_TF32", "BROADCAST_Y",
    "num_buffers_warp_spec", "num_consumer_groups",
    "reg_dec_producer", "reg_inc_consumer",
}


def _to_cc_config(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a flat config dict (our internal format) into the
    ``compare_configs`` recommended-config schema where every non-meta
    constexpr lives inside ``block_size``."""
    bs: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    for k, v in flat.items():
        if k in _META_KEYS:
            meta[k] = v
        else:
            bs[k] = v
    meta["block_size"] = bs
    return meta


def _load_baked_in_config(shapes_file: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(shapes_file.read_text())
    except Exception:
        return None
    kw = data.get("kwargs", {})
    cfg: Dict[str, Any] = {}
    for k, v in kw.items():
        if k in _EXCLUDE_FROM_BAKED_IN:
            continue
        if isinstance(v, dict) and v.get("value_type") == "int":
            cfg[k] = v.get("value")
    return cfg if cfg else None


import re as _re

_LOC_PATTERN = _re.compile(r'loc\("([^"]+\.py)":\d+:\d+\)')
_LIB_MARKERS = ("site-packages", "dist-packages", "/lib/python")


# Handwritten kernels in the run-list whose Python source lives under
# tb/submodules/ rather than extract/ir_output/. Keyed by kernel name.
_HANDWRITTEN_SOURCES: Dict[str, Path] = {
    "_addmm_fwd": _PROJECT_ROOT / "tb" / "submodules" /
        "generative-recommenders" / "generative_recommenders" / "ops" /
        "triton" / "triton_addmm.py",
    # NOTE: there are two confusingly-named "triton_layer_norm.py" files
    # in the tb tree.  The actual definition of
    # ``_layer_norm_fwd_fused_no_bias`` lives in
    # tb/tritonbench/operators/layer_norm/fused_triton.py:9 (matches the
    # TTGIR ``loc()`` directive).  The submodule path that was here
    # before pointed to a file that defines ``_layer_norm_fwd`` (no
    # ``_no_bias`` variant), which made every trial fail with
    # ``module has no attribute _layer_norm_fwd_fused_no_bias``.
    "_layer_norm_fwd_fused_no_bias": _PROJECT_ROOT / "tb" / "tritonbench" /
        "operators" / "layer_norm" / "fused_triton.py",
}


def _resolve_handwritten_py(ir_dir: Path, operator: str, kernel: str) -> Optional[Path]:
    """Parse TTGIR ``loc(...)`` lines to find a handwritten kernel's
    Python source, e.g.
    ``tb/submodules/generative-recommenders/.../triton_addmm.py``."""
    ttgir_candidates = sorted((ir_dir / operator).glob(f"*{kernel}*.ttgir"))
    if not ttgir_candidates:
        return None
    try:
        content = ttgir_candidates[0].read_text()
    except OSError:
        return None
    counts: Dict[str, int] = {}
    for p in _LOC_PATTERN.findall(content):
        if any(m in p for m in _LIB_MARKERS):
            continue
        counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    # Candidate may be absolute — if so and exists, return.
    candidate = Path(best)
    if candidate.exists():
        return candidate
    # Also try relative-to-project-root
    alt = _PROJECT_ROOT / best.lstrip("/")
    return alt if alt.exists() else None


def _find_kernel_py(
    ir_dir: Path, operator: str, kernel: str, shape_idx: int,
) -> Optional[Path]:
    per_input = ir_dir / operator / f"{kernel}_input{shape_idx}.py"
    if per_input.exists():
        return per_input
    shared = ir_dir / operator / f"{kernel}.py"
    if shared.exists():
        return shared
    # Hardcoded fallbacks for handwritten kernels whose source lives
    # outside extract/ir_output/ (typically tb/submodules/).
    hc = _HANDWRITTEN_SOURCES.get(kernel)
    if hc and hc.exists():
        return hc
    # Last-chance: follow TTGIR loc() back to the source tree.
    return _resolve_handwritten_py(ir_dir, operator, kernel)


def _enumerate_search_space(space: Dict[str, List]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*[space[k] for k in keys])
    ]


def _build_config_list(
    operator: str, kernel: str,
    has_autotune: bool, force_full_space: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    if has_autotune and not force_full_space:
        cfgs = get_autotune_configs(operator, kernel)
        if cfgs:
            return (list(cfgs), "autotune_decorator")
    space = get_search_space(operator, kernel)
    if not space:
        return ([], "empty")
    return (_enumerate_search_space(space), "common_search_space")


def _shape_idx_from_path(p: str) -> Optional[int]:
    name = Path(p).stem  # ..._input<N>_shapes
    m = name.split("_input")
    if len(m) < 2:
        return None
    try:
        return int(m[-1].split("_")[0])
    except ValueError:
        return None


def run_operator_cc(
    operator: str,
    kernel: str,
    output_path: Path,
    ir_dir: Path,
    has_autotune: bool = False,
    device: str = "cuda",
    warmup: int = 5,
    rep: int = 20,
    force_full_space: bool = False,
) -> List[Dict[str, Any]]:
    """Grid-search every shape via compare_configs harness; persist per trial."""
    import torch
    from compare_configs import benchmark_config_pair

    shape_files = sorted(glob.glob(
        str(ir_dir / operator / f"*_{kernel}_input*_shapes.json")
    ))
    if not shape_files:
        log.error("No shapes_files for %s/%s under %s", operator, kernel, ir_dir)
        return [_error_shape(operator, kernel, -1, "no shapes files")]

    effective_has_autotune = (
        has_autotune and get_autotune_configs(operator, kernel) is not None
    )
    config_list, config_source = _build_config_list(
        operator, kernel, effective_has_autotune,
        force_full_space=force_full_space,
    )
    n_configs = len(config_list)
    log.info(
        "Grid-searching %d configs/shape over %d shapes (source=%s, "
        "has_autotune=%s -> effective=%s)",
        n_configs, len(shape_files), config_source,
        has_autotune, effective_has_autotune,
    )
    if n_configs == 0:
        return [_error_shape(operator, kernel, -1, "empty config list")]

    existing, done_trials = _load_partial(output_path)
    results: List[Dict[str, Any]] = list(existing)
    existing_by_id: Dict[int, Dict[str, Any]] = {
        int(s["input_id"]): s for s in existing if s.get("input_id") is not None
    }

    def _flush(status: str = "in_progress"):
        _flush_envelope(
            output_path, operator, kernel, results,
            has_autotune=has_autotune, status=status,
        )

    def _finalize_shape(rec: Dict[str, Any], n_configs: int) -> None:
        best_lat = None
        best_cfg = None
        best_id = None
        for t in rec.get("trial_history", []):
            lat = t.get("latency_ms")
            if lat is None:
                continue
            if best_lat is None or lat < best_lat:
                best_lat, best_cfg, best_id = lat, t.get("config"), t.get("trial_id")
        rec["best_config"] = best_cfg
        rec["best_latency_ms"] = best_lat
        rec["latency_ms"] = best_lat
        rec["best_trial_id"] = best_id
        orig_lat = rec.get("original_config_latency_ms")
        if orig_lat is not None and best_lat and best_lat > 0:
            rec["speedup"] = orig_lat / best_lat
        if best_cfg is not None:
            rec["status"] = "success"
            rec["verified"] = True
            rec["error"] = None
        else:
            rec["status"] = "error"
            rec["verified"] = False
            rec["error"] = f"All {n_configs} trials failed"

    # Finalize any shapes loaded from partial results so the per-kernel
    # JSON always has best_latency_ms populated even if the previous run
    # exited mid-shape.
    for rec in results:
        if rec.get("trial_history") and rec.get("best_latency_ms") is None:
            _finalize_shape(rec, rec.get("n_configs", n_configs))
    _flush()

    for shape_file in shape_files:
        shape_idx = _shape_idx_from_path(shape_file)
        if shape_idx is None:
            continue

        shapes_path = Path(shape_file)
        baked_cfg = _load_baked_in_config(shapes_path)
        if baked_cfg is None:
            log.warning("shape %d: missing baked-in config", shape_idx)
            continue

        kernel_py = _find_kernel_py(ir_dir, operator, kernel, shape_idx)
        if kernel_py is None:
            log.warning("shape %d: missing kernel .py", shape_idx)
            continue

        # For template kernels (``@triton_heuristics.template``) block
        # sizes are hardcoded in the source, not in the shapes_file
        # kwargs. Inject them into baked_cfg so compare_configs'
        # grid-resolution and source-patching path computes the same
        # grid Full used.
        try:
            _src = kernel_py.read_text()
        except OSError:
            _src = ""
        from compare_configs import (  # noqa: E402
            _extract_hardcoded_constexprs as _cc_extract_constexprs,
            _is_template_kernel as _cc_is_template,
        )
        if _src and _cc_is_template(_src):
            hc = _cc_extract_constexprs(_src)
            for k, v in hc.items():
                baked_cfg.setdefault(k, v)
        baked_cc = _to_cc_config(baked_cfg)

        rec = existing_by_id.get(shape_idx)
        if rec is None:
            rec = {
                "operator": operator,
                "kernel": kernel,
                "input_id": shape_idx,
                "x_val": str(shape_idx),
                "has_autotune": has_autotune,
                "triton_impl_name": None,
                "n_configs": n_configs,
                "config_source": config_source,
                "reference_config": copy.deepcopy(baked_cfg),
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
                "harness": "compare_configs",
            }
            results.append(rec)
            existing_by_id[shape_idx] = rec
        else:
            rec["n_configs"] = n_configs
            rec["config_source"] = config_source
            rec.setdefault("harness", "compare_configs")

        for trial_id, cfg in enumerate(config_list):
            if (shape_idx, trial_id) in done_trials:
                continue
            rec_cc = _to_cc_config(cfg)

            try:
                r = benchmark_config_pair(
                    original_config=baked_cc,
                    recommended_config=rec_cc,
                    shapes_file=shapes_path,
                    kernel_module=kernel_py,
                    kernel_name=kernel,
                    device=device,
                    warmup=warmup, rep=rep, seed=42,
                    validate_outputs=False,
                )
                orig_stats = r.get("original_stats")
                if (orig_stats
                        and "error" not in orig_stats
                        and rec.get("original_config_latency_ms") is None):
                    rec["original_stats"] = orig_stats
                    rec["original_config_latency_ms"] = orig_stats.get("median")

                rec_stats = r.get("recommended_stats") or {}
                if "error" in rec_stats or rec_stats.get("median") is None:
                    trial_status = "error"
                    trial_error = rec_stats.get("error", "no median")
                    lat = None
                    rec["trial_errors"] += 1
                else:
                    lat = rec_stats.get("median")
                    trial_status = "success"
                    trial_error = None
            except Exception as exc:
                lat = None
                trial_status = "error"
                trial_error = str(exc)
                rec["trial_errors"] += 1

            rec["trial_history"].append({
                "trial_id": trial_id,
                "config": copy.deepcopy(cfg),
                "latency_ms": lat,
                "status": trial_status,
                "error": trial_error,
            })
            done_trials.add((shape_idx, trial_id))
            _flush()

            # A CUDA "illegal memory access" poisons the context for the
            # rest of the process; every subsequent torch call raises the
            # same error. Exit so the orchestrator can re-launch us with
            # a fresh CUDA context (per-trial flush keeps progress safe).
            if trial_error and (
                "illegal memory access" in trial_error
                or "CUDA error" in trial_error
            ):
                log.warning(
                    "CUDA context corrupted after trial %d — exiting for "
                    "orchestrator restart.", trial_id,
                )
                _finalize_shape(rec, n_configs)
                _flush()
                sys.exit(2)

        _finalize_shape(rec, n_configs)
        _flush()

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


def main():
    parser = argparse.ArgumentParser(
        description="Autotune grid-search via compare_configs harness",
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--has-autotune", action="store_true", default=False)
    parser.add_argument("--full-space", action="store_true", default=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.time()
    try:
        results = run_operator_cc(
            operator=args.operator,
            kernel=args.kernel,
            output_path=args.output.resolve(),
            ir_dir=args.ir_dir.resolve(),
            has_autotune=args.has_autotune,
            device=args.device,
            warmup=args.warmup, rep=args.rep,
            force_full_space=args.full_space,
        )
    except Exception as exc:
        log.error("Fatal: %s", exc)
        traceback.print_exc()
        results = [_error_shape(args.operator, args.kernel, -1, f"Fatal: {exc}")]
        _flush_envelope(
            args.output.resolve(), args.operator, args.kernel, results,
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
