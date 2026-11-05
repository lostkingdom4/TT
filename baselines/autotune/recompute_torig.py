"""Recompute per-shape ``t_orig`` against the kernel's **baked-in** config.

The paper's multi-agent pipeline (``optimizer_functions.py:699–744``) uses
the config that was compiled into the kernel's TTGIR / source at
IR-extraction time as its ``t_orig``. This config lives in
``extract/ir_output/<device>/<op>/<kernel>_input<N>_shapes.json`` under
``kwargs``. For ``@triton.autotune``-wrapped kernels it's the autotuner's
winner for the extraction shape; for Inductor-codegen kernels it's
whatever Inductor chose.

Our grid-search baseline originally measured ``t_orig`` against the
first entry in its search list (a deliberately bad config), which
inflated speedup ratios for GEMMs. This script re-measures ``t_orig``
against the baked-in config per shape and writes two new fields into
the existing per-kernel JSONs:

    baked_in_config        — dict per shape
    baked_in_latency_ms    — do_bench median of that config
    speedup_vs_baked_in    — baked_in_latency_ms / best_latency_ms

The original fields (``original_config_latency_ms``, ``best_latency_ms``,
``speedup``) are **not** modified so downstream consumers can pick the
convention they want.

Usage:

    # One kernel (direct):
    python -m baselines.autotune.recompute_torig \\
        --operator addmm --kernel _addmm_fwd \\
        --json baselines/results/v100/autotune/addmm___addmm_fwd.json \\
        --ir-dir extract/ir_output/v100

    # All 19 kernels (orchestrator):
    python -m baselines.autotune.recompute_torig \\
        --run-list run_lists/v100_improved_only.json \\
        --results-dir baselines/results/v100/autotune \\
        --ir-dir extract/ir_output/v100
"""

import argparse
import copy
import gc
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.autotune.operator_runner import (
    _bench_once,
    _find_triton_impl,
    _flush_envelope,
)
from baselines.common.injector import _FixedConfigInjector
from baselines.common.tritonbench_setup import (
    build_tb_args,
    fresh_cache_context,
    load_op_class,
)

log = logging.getLogger(__name__)


# Scalar kwargs that are semantically non-tunable (bools, internal
# accounting). We exclude them from the injected baked-in config.
_EXCLUDE_FROM_BAKED_IN = {
    "warmup", "ALLOW_TF32", "BROADCAST_Y",
    "num_buffers_warp_spec", "num_consumer_groups",
    "reg_dec_producer", "reg_inc_consumer",
}


def _load_baked_in_configs(
    ir_dir: Path, operator: str, kernel: str,
) -> Dict[int, Dict[str, Any]]:
    """Return ``{input_idx: baked_in_config_dict}`` for one kernel.

    Each config is the int-valued ``kwargs`` from
    ``<op>/<op>_<kernel>_input<N>_shapes.json`` with semantically
    non-tunable fields dropped.
    """
    pattern = str(ir_dir / operator / f"*_{kernel}_input*_shapes.json")
    files = sorted(glob.glob(pattern))
    out: Dict[int, Dict[str, Any]] = {}
    for p in files:
        # Extract input index from filename: ..._input<N>_shapes.json
        name = Path(p).stem
        idx_marker = "_input"
        if idx_marker not in name:
            continue
        try:
            idx = int(name.split(idx_marker)[-1].split("_")[0])
        except ValueError:
            continue
        try:
            data = json.loads(Path(p).read_text())
        except Exception:
            continue
        kw = data.get("kwargs", {})
        cfg: Dict[str, Any] = {}
        for k, v in kw.items():
            if k in _EXCLUDE_FROM_BAKED_IN:
                continue
            if isinstance(v, dict) and v.get("value_type") == "int":
                cfg[k] = v.get("value")
        out[idx] = cfg
    return out


def recompute_one_kernel(
    operator: str,
    kernel: str,
    json_path: Path,
    ir_dir: Path,
    has_autotune: bool,
    device: str = "cuda",
    warmup: int = 5,
    rep: int = 20,
) -> Dict[str, Any]:
    """Re-measure ``t_orig`` for every shape of one kernel's JSON."""
    import torch

    # Load existing result JSON
    data = json.loads(json_path.read_text())
    shapes: List[Dict[str, Any]] = data.get("shapes", [])
    if not shapes:
        return {"status": "no_shapes", "kernel": kernel}

    baked = _load_baked_in_configs(ir_dir, operator, kernel)
    if not baked:
        return {"status": "no_shapes_json", "kernel": kernel}

    tb_args, extra_args = build_tb_args(
        device=device, precision="fp16",
        metrics="latency",
        warmup=warmup, rep=rep,
    )
    OpClass = load_op_class(operator)

    n_measured = 0
    n_missing_cfg = 0
    n_bench_fail = 0

    with fresh_cache_context():
        op = OpClass(tb_args=tb_args, extra_args=extra_args)
        triton_impl = _find_triton_impl(op)
        if triton_impl is None:
            return {"status": "no_triton_impl", "kernel": kernel}
        impl_method = getattr(op, triton_impl, None)
        if impl_method is None:
            return {"status": "no_impl_method", "kernel": kernel}

        shape_idx = 0
        while True:
            try:
                example_inputs = op.get_example_inputs()
            except Exception as exc:
                log.error("get_example_inputs() failed at %d: %s", shape_idx, exc)
                break
            if example_inputs is None:
                break

            # Find the matching shape_rec in our results
            rec = None
            for s in shapes:
                if s.get("input_id") == shape_idx:
                    rec = s; break

            baked_cfg = baked.get(shape_idx)
            if baked_cfg is None:
                n_missing_cfg += 1
                if rec is not None:
                    rec["baked_in_config"] = None
                    rec["baked_in_latency_ms"] = None
                    rec["speedup_vs_baked_in"] = None
                shape_idx += 1
                continue

            # Benchmark baked-in config under the injector
            try:
                with _FixedConfigInjector(
                    operator, baked_cfg, has_autotune, kernel=kernel,
                ):
                    stats = _bench_once(impl_method, example_inputs,
                                        warmup=warmup, rep=rep)
                baked_lat = stats.get("median") if stats else None
            except Exception as exc:
                log.debug("baked-in bench failed for shape %d: %s", shape_idx, exc)
                baked_lat = None

            if baked_lat is None:
                n_bench_fail += 1

            if rec is not None:
                rec["baked_in_config"] = copy.deepcopy(baked_cfg)
                rec["baked_in_latency_ms"] = baked_lat
                best = rec.get("best_latency_ms")
                if baked_lat is not None and best is not None and best > 0:
                    rec["speedup_vs_baked_in"] = baked_lat / best
                else:
                    rec["speedup_vs_baked_in"] = None
                n_measured += 1
            shape_idx += 1

    # Write back atomically
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, json_path)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    return {
        "status": "ok",
        "kernel": f"{operator}/{kernel}",
        "shapes_measured": n_measured,
        "shapes_missing_cfg": n_missing_cfg,
        "shapes_bench_failed": n_bench_fail,
    }


def _run_subprocess(
    operator: str, kernel: str, json_path: Path, ir_dir: Path,
    has_autotune: bool, timeout: int, log_level: str,
) -> int:
    cmd = [
        sys.executable, "-m", "baselines.autotune.recompute_torig",
        "--operator", operator, "--kernel", kernel,
        "--json", str(json_path), "--ir-dir", str(ir_dir),
        "--log-level", log_level,
    ]
    if has_autotune:
        cmd.append("--has-autotune")
    try:
        proc = subprocess.run(
            cmd, timeout=timeout, capture_output=True, text=True,
            cwd=str(_PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)},
        )
        if proc.stdout.strip():
            for line in proc.stdout.strip().splitlines()[-8:]:
                print(f"  [{operator}/{kernel}] {line}")
        return proc.returncode
    except subprocess.TimeoutExpired:
        log.error("Timeout (%ds) for %s/%s", timeout, operator, kernel)
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Recompute per-shape t_orig against the kernel's "
                    "baked-in config (matches paper/Full convention).",
    )
    parser.add_argument("--operator", default=None,
                        help="If set: run only this one (op, kernel) directly.")
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--json", type=Path, default=None,
                        help="Per-kernel result JSON to update "
                             "(required when --operator/--kernel is set).")
    parser.add_argument("--ir-dir", type=Path,
                        default=_PROJECT_ROOT / "extract" / "ir_output" / "v100",
                        help="IR output dir containing per-shape JSONs.")
    parser.add_argument("--results-dir", type=Path,
                        default=_PROJECT_ROOT / "baselines" / "results" / "v100" / "autotune",
                        help="Autotune results dir (orchestrator mode).")
    parser.add_argument("--run-list", type=Path,
                        default=_PROJECT_ROOT / "run_lists" / "v100_improved_only.json",
                        help="Run-list JSON (orchestrator mode).")
    parser.add_argument("--has-autotune", action="store_true", default=False,
                        help="Direct mode: whether the kernel uses @triton.autotune.")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-kernel subprocess timeout (orchestrator mode).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Direct mode: one kernel
    if args.operator and args.kernel and args.json:
        t0 = time.time()
        result = recompute_one_kernel(
            operator=args.operator, kernel=args.kernel,
            json_path=args.json.resolve(), ir_dir=args.ir_dir.resolve(),
            has_autotune=args.has_autotune,
        )
        elapsed = time.time() - t0
        print(f"[{args.operator}/{args.kernel}] {result} "
              f"(elapsed {elapsed:.1f}s)")
        return

    # Orchestrator mode: iterate run-list, subprocess per kernel
    from baselines.common.discover import load_run_list
    from baselines.common.search_space import get_autotune_configs

    rl = load_run_list(args.run_list)
    print(f"Recomputing t_orig for {len(rl)} kernels from {args.run_list.name}")

    done = fail = skip = 0
    for i, (op, kern) in enumerate(rl, 1):
        # JSON filename: operator__kernel.json, with double underscore when
        # kernel starts with _
        fname = f"{op}__{kern}.json"
        if kern.startswith("_"):
            fname = f"{op}___{kern[1:]}.json"
        jpath = args.results_dir / fname
        if not jpath.exists():
            print(f"[{i}/{len(rl)}] {op}/{kern}: JSON MISSING {jpath}")
            skip += 1; continue

        has_autotune = get_autotune_configs(op, kern) is not None
        print(f"[{i}/{len(rl)}] {op}/{kern}  (has_autotune={has_autotune})")
        rc = _run_subprocess(
            op, kern, jpath, args.ir_dir, has_autotune,
            timeout=args.timeout, log_level=args.log_level,
        )
        if rc == 0:
            done += 1
        else:
            fail += 1
            print(f"  subprocess exit {rc}")

    print(f"\nDone: {done} kernels recomputed, {fail} failed, {skip} skipped")


if __name__ == "__main__":
    main()
