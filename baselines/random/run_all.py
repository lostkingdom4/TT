#!/usr/bin/env python3
"""CLI entry point for the random-config baseline.

Discovers operators from an IR output directory (optionally filtered by a
run-list JSON), runs each one through TritonBench with randomly-generated
kernel configurations, verifies output correctness, and records chosen
configs together with latency measurements.

Each operator runs in a separate subprocess for GPU isolation.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.common.discover import (  # noqa: E402
    discover_kernels,
    discover_operators,
    filter_by_run_list,
    get_device_name,
    load_run_list,
    print_discovery_summary,
)
from baselines.common.search_space import get_search_space  # noqa: E402
from baselines.random.results import RandomConfigResultsStore  # noqa: E402

log = logging.getLogger(__name__)


def _run_operator_subprocess(
    operator: str,
    kernel: str,
    has_autotune: bool,
    seed: int,
    device: str,
    warmup: int,
    rep: int,
    timeout: int,
    log_level: str,
) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix=f"rcb_{operator}_{kernel}_",
        delete=False, mode="w",
    ) as tmp:
        output_path = tmp.name

    cmd = [
        sys.executable, "-m", "baselines.random.operator_runner",
        "--operator", operator,
        "--seed", str(seed),
        "--device", device,
        "--warmup", str(warmup),
        "--rep", str(rep),
        "--output", output_path,
        "--log-level", log_level,
    ]
    if kernel:
        cmd.extend(["--kernel", kernel])
    if has_autotune:
        cmd.append("--has-autotune")

    log.info("Spawning: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, timeout=timeout, capture_output=True, text=True,
            cwd=str(_PROJECT_ROOT),
            env={**os.environ,
                 "PYTHONPATH": str(_PROJECT_ROOT),
                 "TRITON_PRINT_AUTOTUNING": "1"},
        )

        for line in (proc.stdout or "").strip().splitlines():
            if line:
                print(f"  [{operator}] {line}")

        if proc.returncode != 0:
            log.warning("Subprocess for %s exited with code %d",
                        operator, proc.returncode)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path) as f:
                return json.load(f)
        return [_err(operator, kernel, has_autotune, seed,
                     f"No output file (exit code {proc.returncode})")]

    except subprocess.TimeoutExpired:
        log.error("Timeout (%ds) for %s", timeout, operator)
        return [_err(operator, kernel, has_autotune, seed,
                     f"Subprocess timed out after {timeout}s")]
    except Exception as exc:
        log.error("Subprocess error for %s: %s", operator, exc)
        return [_err(operator, kernel, has_autotune, seed, str(exc))]
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _err(operator, kernel, has_autotune, seed, error):
    return {
        "operator": operator, "kernel": kernel,
        "input_id": -1, "x_val": "N/A",
        "random_seed": seed, "has_autotune": has_autotune,
        "random_config": None, "latency_ms": None,
        "verified": False, "verification_error": error,
        "triton_impl_name": None, "status": "error", "error": error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run random-config baseline for all operators",
    )
    parser.add_argument("--ir-dir", type=Path, required=True,
                        help="IR output directory (e.g. extract/ir_output/v100)")
    parser.add_argument("--run-list", type=Path, default=None,
                        help="Optional run-list JSON to filter (operator, kernel) pairs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Results output directory (default: baselines/results)")
    parser.add_argument("--operator", type=str, default=None,
                        help="Run only this operator")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Run only this kernel (used with --operator)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--export-csv", action="store_true", default=False)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ir_dir = args.ir_dir.resolve()
    device = get_device_name(ir_dir)
    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else _PROJECT_ROOT / "baselines" / "results"
    )

    print(f"\nRandom Config Baseline Runner")
    print(f"  IR directory:   {ir_dir}")
    print(f"  Device:         {device}")
    print(f"  Output:         {output_dir}")
    print(f"  Run-list:       {args.run_list}")
    print(f"  Random seed:    {args.seed}")
    print(f"  Resume:         {args.resume}")
    print(f"  Warmup/Rep:     {args.warmup}/{args.rep}")
    print(f"  Timeout:        {args.timeout}s per operator")

    # ---- Discovery ----
    if args.run_list:
        kernels = discover_kernels(ir_dir)
        run_list = load_run_list(args.run_list)
        entries = filter_by_run_list(kernels, run_list)
        if args.operator:
            entries = [e for e in entries if e["operator"] == args.operator]
        if args.kernel:
            entries = [e for e in entries if e["kernel"] == args.kernel]
    else:
        operators = discover_operators(ir_dir)
        tb_operators = [op for op in operators if op["in_tritonbench"]]
        if args.operator:
            tb_operators = [op for op in tb_operators if op["name"] == args.operator]
        # Promote to per-kernel entries using the first shapes file's kernel name
        entries = [{
            "operator": op["name"],
            "kernel": op["kernel_name"],
            "num_shapes": op["num_shapes"],
            "has_autotune": op["has_autotune"],
            "in_tritonbench": op["in_tritonbench"],
        } for op in tb_operators]

    if not entries:
        print("\nNo kernels to run.")
        return

    print_discovery_summary(entries, device)

    print(f"  Search spaces:")
    for e in entries:
        space = get_search_space(e["operator"], e.get("kernel"))
        print(f"    {e['operator']}/{e.get('kernel','')}: "
              f"{', '.join(f'{k}={v}' for k,v in space.items())}")
    print()

    store = RandomConfigResultsStore(output_dir, device, seed=args.seed)

    completed, failed, skipped = 0, 0, 0
    total = len(entries)

    for i, e in enumerate(entries, 1):
        op_name = e["operator"]; kernel = e.get("kernel")
        has_autotune = e["has_autotune"]
        num_shapes = e.get("num_shapes", "?")

        if args.resume and store.is_completed(op_name, kernel):
            print(f"\n[{i}/{total}] {op_name}/{kernel}: SKIPPED (already completed)")
            skipped += 1
            continue

        autotune_tag = "autotune" if has_autotune else "no-autotune"
        print(f"\n[{i}/{total}] {op_name}/{kernel} "
              f"({num_shapes} shapes, {autotune_tag}, seed={args.seed})")

        t0 = time.time()
        results = _run_operator_subprocess(
            operator=op_name, kernel=kernel,
            has_autotune=has_autotune, seed=args.seed,
            device="cuda", warmup=args.warmup, rep=args.rep,
            timeout=args.timeout, log_level=args.log_level,
        )
        elapsed = time.time() - t0

        ok = sum(1 for r in results if r.get("status") == "success")
        err = sum(1 for r in results if r.get("status") == "error")
        corr = sum(1 for r in results if r.get("status") == "correctness_error")
        verified = sum(1 for r in results if r.get("verified"))
        with_cfg = sum(1 for r in results if r.get("random_config"))

        if ok > 0:
            status = "completed"
            completed += 1
        else:
            status = "failed"
            failed += 1

        store.save_operator_results(
            operator=op_name, kernel=kernel,
            shape_results=results,
            status=status,
            extra={"has_autotune": has_autotune, "random_seed": args.seed},
        )

        print(f"  Done in {elapsed:.1f}s — shapes={len(results)}, "
              f"ok={ok}, errors={err}, corr_errors={corr}, "
              f"verified={verified}, with_config={with_cfg}")

    # ---- Summary ----
    print(f"\n{'=' * 70}\n  FINAL SUMMARY\n{'=' * 70}")
    print(f"  Total kernels:   {total}")
    print(f"  Completed:       {completed}")
    print(f"  Failed:          {failed}")
    print(f"  Skipped (resume):{skipped}")
    store.print_summary()
    if args.export_csv:
        csv_path = store.export_csv()
        print(f"  CSV exported to: {csv_path}")
    summary_path = store.device_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(store.export_summary(), f, indent=2, default=str)
    print(f"  Summary JSON:    {summary_path}")


if __name__ == "__main__":
    main()
