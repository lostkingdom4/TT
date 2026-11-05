#!/usr/bin/env python3
"""CLI entry point for the exhaustive grid-search autotuning baseline.

Discovers (operator, kernel) pairs from a run-list JSON
(default: ``run_lists/v100_improved_only.json``) and for each pair spawns
a subprocess that:

* for kernels wrapped with ``@triton.autotune``: enumerates the kernel's
  own decorator ``configs=[...]`` list (the "default search space");
* for non-autotuned kernels: enumerates the full Cartesian product of
  ``baselines.common.search_space.get_search_space(operator, kernel)``;

and benchmarks every config with the same methodology as
``batch_runner`` (``do_bench(warmup=5, rep=20)`` + L2 flush).

The subprocess writes its JSON output **atomically after every trial**,
so interrupt-and-resume never loses more than one trial's worth of work.

Usage examples::

    # Full run over run_lists/v100_improved_only.json
    python -m baselines.autotune.run_all \\
        --ir-dir extract/ir_output/v100 --export-csv

    # Single kernel
    python -m baselines.autotune.run_all \\
        --ir-dir extract/ir_output/v100 \\
        --operator rope --kernel triton_poi_fused_add_cat_mul_0

    # Resume (default)
    python -m baselines.autotune.run_all --ir-dir extract/ir_output/v100

    # Force rerun
    python -m baselines.autotune.run_all --ir-dir extract/ir_output/v100 \\
        --no-resume
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.autotune.results import BaselineResultsStore  # noqa: E402
from baselines.common.discover import (  # noqa: E402
    discover_kernels,
    filter_by_run_list,
    get_device_name,
    load_run_list,
    print_discovery_summary,
)
from baselines.common.search_space import get_search_space  # noqa: E402

_DEFAULT_RUN_LIST = _PROJECT_ROOT / "run_lists" / "v100_improved_only.json"

log = logging.getLogger(__name__)


def _run_subprocess(
    operator: str,
    kernel: str,
    has_autotune: bool,
    output_path: Path,
    device: str,
    warmup: int,
    rep: int,
    timeout: int,
    log_level: str,
) -> int:
    """Spawn the subprocess that does the grid-search for one kernel.

    The subprocess writes directly to ``output_path`` (atomically, after
    every trial).  Returns the subprocess exit code.
    """
    cmd = [
        sys.executable, "-m", "baselines.autotune.operator_runner",
        "--operator", operator,
        "--kernel", kernel,
        "--device", device,
        "--warmup", str(warmup),
        "--rep", str(rep),
        "--output", str(output_path),
        "--log-level", log_level,
    ]
    if has_autotune:
        cmd.append("--has-autotune")

    log.info("Spawning: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, timeout=timeout, capture_output=True, text=True,
            cwd=str(_PROJECT_ROOT),
            env={**os.environ,
                 "PYTHONPATH": str(_PROJECT_ROOT),
                 "TRITON_PRINT_AUTOTUNING": "0"},
        )
        for line in (proc.stdout or "").strip().splitlines():
            if line:
                print(f"  [{operator}/{kernel}] {line}")
        if proc.stderr.strip():
            for line in proc.stderr.strip().splitlines():
                log.debug("[%s/%s stderr] %s", operator, kernel, line)
        if proc.returncode != 0:
            log.warning("Subprocess for %s/%s exited with code %d",
                        operator, kernel, proc.returncode)
        return proc.returncode
    except subprocess.TimeoutExpired:
        log.error("Timeout (%ds) for %s/%s", timeout, operator, kernel)
        return -1
    except Exception as exc:
        log.error("Subprocess error for %s/%s: %s", operator, kernel, exc)
        return -2


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive grid-search autotuning baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ir-dir", type=Path, required=True,
                        help="IR output directory (e.g. extract/ir_output/v100)")
    parser.add_argument("--run-list", type=Path, default=_DEFAULT_RUN_LIST,
                        help=f"Run-list JSON (default: {_DEFAULT_RUN_LIST.name})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Results output directory (default: baselines/results)")
    parser.add_argument("--operator", type=str, default=None,
                        help="Run only this operator (optional filter)")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Run only this kernel (optional filter)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="do_bench warmup (default: 5, matches batch_runner)")
    parser.add_argument("--rep", type=int, default=20,
                        help="do_bench rep (default: 20, matches batch_runner)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-completed kernels (default: True). "
                             "Within a kernel, partial trial history in the "
                             "per-kernel JSON is always honoured.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-run every kernel from scratch (deletes "
                             "existing per-kernel JSON files)")
    parser.add_argument("--timeout", type=int, default=14400,
                        help="Per-kernel subprocess timeout in seconds "
                             "(default: 14400 = 4 h — big search spaces on "
                             "GEMM-like kernels can take a while)")
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

    print(f"\nExhaustive Grid-Search Autotuning Baseline")
    print(f"  IR directory:   {ir_dir}")
    print(f"  Device:         {device}")
    print(f"  Output:         {output_dir}")
    print(f"  Run-list:       {args.run_list}")
    print(f"  Warmup/Rep:     {args.warmup}/{args.rep}")
    print(f"  Resume:         {args.resume}")
    print(f"  Timeout:        {args.timeout}s per kernel")

    # ---- Discovery ----
    kernels = discover_kernels(ir_dir)
    run_list = load_run_list(args.run_list)
    entries = filter_by_run_list(kernels, run_list)

    if args.operator:
        entries = [e for e in entries if e["operator"] == args.operator]
    if args.kernel:
        entries = [e for e in entries if e["kernel"] == args.kernel]

    if not entries:
        print("\nNo (operator, kernel) pairs to run.")
        return

    print_discovery_summary(entries, device)

    print(f"  Search spaces (non-autotuned kernels only):")
    for e in entries:
        if e.get("has_autotune"):
            print(f"    {e['operator']}/{e['kernel']}: <captured from @triton.autotune>")
            continue
        space = get_search_space(e["operator"], e["kernel"])
        n_configs = 1
        for v in space.values():
            n_configs *= len(v)
        print(f"    {e['operator']}/{e['kernel']}: "
              f"{n_configs} configs = " +
              ", ".join(f"{k}({len(v)})" for k, v in space.items()))
    print()

    store = BaselineResultsStore(output_dir, device)

    # If --no-resume, wipe existing per-kernel JSONs for the entries we're
    # about to process so the subprocess starts from scratch.
    if not args.resume:
        for e in entries:
            p = store.operator_file_path(e["operator"], e["kernel"])
            if p.exists():
                log.info("Deleting %s (--no-resume)", p)
                p.unlink()

    completed, failed, skipped = 0, 0, 0
    total = len(entries)

    for i, e in enumerate(entries, 1):
        op_name = e["operator"]
        kernel = e["kernel"]
        has_autotune = e.get("has_autotune", False)
        num_shapes = e.get("num_shapes", "?")

        if args.resume and store.is_completed(op_name, kernel):
            print(f"\n[{i}/{total}] {op_name}/{kernel}: "
                  f"SKIPPED (already completed)")
            skipped += 1
            continue

        tag = "autotune" if has_autotune else "no-autotune"
        print(f"\n[{i}/{total}] {op_name}/{kernel} "
              f"({num_shapes} shapes, {tag})")

        output_path = store.operator_file_path(op_name, kernel)

        t0 = time.time()
        rc = _run_subprocess(
            operator=op_name, kernel=kernel, has_autotune=has_autotune,
            output_path=output_path,
            device="cuda", warmup=args.warmup, rep=args.rep,
            timeout=args.timeout, log_level=args.log_level,
        )
        elapsed = time.time() - t0

        # Read back whatever the subprocess produced (partial or full) and
        # roll it into the store so index.json reflects reality.
        envelope = store.load_operator_results(op_name, kernel)
        shape_results = envelope.get("shapes", []) if envelope else []
        ok = sum(1 for r in shape_results if r.get("status") == "success")
        err = sum(1 for r in shape_results if r.get("status") == "error")
        improved = sum(1 for r in shape_results
                       if r.get("speedup") and r["speedup"] > 1.005)

        if rc == 0 and ok > 0:
            status = "completed"
            completed += 1
        else:
            status = "failed"
            failed += 1

        store.save_operator_results(
            operator=op_name, kernel=kernel,
            shape_results=shape_results,
            status=status,
            extra={"has_autotune": has_autotune},
        )

        print(f"  Done in {elapsed:.1f}s — shapes={len(shape_results)}, "
              f"ok={ok}, errors={err}, improved={improved}")

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
