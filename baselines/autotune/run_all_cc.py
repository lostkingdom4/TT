#!/usr/bin/env python3
"""Run the compare_configs-harness autotune baseline over a run-list.

Same orchestration surface as ``baselines/autotune/run_all.py`` but
spawns ``operator_runner_cc.py`` subprocesses. Results are written to
``<output-dir>/<device>/autotune/`` with a dedicated
``--output-dir baselines/results_cc/`` by convention so the old
operator-wrapper results stay untouched.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.autotune.results import BaselineResultsStore  # noqa: E402
from baselines.common.discover import (  # noqa: E402
    discover_kernels, filter_by_run_list, get_device_name,
    load_run_list, print_discovery_summary,
)
from baselines.common.search_space import get_search_space  # noqa: E402

_DEFAULT_RUN_LIST = _PROJECT_ROOT / "run_lists" / "v100_improved_only.json"
log = logging.getLogger(__name__)


def _run_subprocess(
    operator: str, kernel: str, has_autotune: bool,
    output_path: Path, ir_dir: Path,
    warmup: int, rep: int, timeout: int, log_level: str,
) -> int:
    cmd = [
        sys.executable, "-m", "baselines.autotune.operator_runner_cc",
        "--operator", operator, "--kernel", kernel,
        "--ir-dir", str(ir_dir),
        "--warmup", str(warmup), "--rep", str(rep),
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
        if proc.returncode != 0:
            log.warning("Exit code %d for %s/%s", proc.returncode, operator, kernel)
            for line in (proc.stderr or "").strip().splitlines()[-10:]:
                log.warning("  stderr: %s", line)
        return proc.returncode
    except subprocess.TimeoutExpired:
        log.error("Timeout (%ds) for %s/%s", timeout, operator, kernel)
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Autotune grid-search baseline via compare_configs harness",
    )
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--run-list", type=Path, default=_DEFAULT_RUN_LIST)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Results root (default: baselines/results_cc)")
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--timeout", type=int, default=14400)
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
        else _PROJECT_ROOT / "baselines" / "results_cc"
    )

    print("\nCompare-configs-harness Autotune Baseline")
    print(f"  IR dir:     {ir_dir}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print(f"  Run-list:   {args.run_list}")
    print(f"  Warmup/Rep: {args.warmup}/{args.rep}")
    print(f"  Resume:     {args.resume}")

    kernels = discover_kernels(ir_dir)
    run_list = load_run_list(args.run_list)
    entries = filter_by_run_list(kernels, run_list)
    if args.operator:
        entries = [e for e in entries if e["operator"] == args.operator]
    if args.kernel:
        entries = [e for e in entries if e["kernel"] == args.kernel]
    if not entries:
        print("\nNo entries to run.")
        return

    print_discovery_summary(entries, device)

    store = BaselineResultsStore(output_dir, device)

    if not args.resume:
        for e in entries:
            p = store.operator_file_path(e["operator"], e["kernel"])
            if p.exists():
                log.info("Deleting %s", p)
                p.unlink()

    total = len(entries)
    completed = failed = skipped = 0
    for i, e in enumerate(entries, 1):
        op = e["operator"]; kn = e["kernel"]
        has_autotune = e.get("has_autotune", False)
        if args.resume and store.is_completed(op, kn):
            print(f"\n[{i}/{total}] {op}/{kn}: SKIPPED (already completed)")
            skipped += 1
            continue

        output_path = store.operator_file_path(op, kn)
        tag = "autotune" if has_autotune else "no-autotune"
        print(f"\n[{i}/{total}] {op}/{kn} ({tag})")

        t0 = time.time()
        # Retry on exit 2 (CUDA corruption) — resume picks up from
        # partial results, so every retry makes progress toward the
        # remaining configs. Count only retries that produced *no*
        # additional trials — a retry that extended progress resets
        # the budget. This handles kernels whose search spaces contain
        # many crashing configs without capping prematurely.
        MAX_NO_PROGRESS = 4
        no_progress = 0
        rc = -1

        def _trials_recorded() -> int:
            env = store.load_operator_results(op, kn) or {}
            n = 0
            for s in env.get("shapes", []):
                n += len(s.get("trial_history", []))
            return n

        while True:
            before = _trials_recorded()
            rc = _run_subprocess(
                operator=op, kernel=kn, has_autotune=has_autotune,
                output_path=output_path, ir_dir=ir_dir,
                warmup=args.warmup, rep=args.rep,
                timeout=args.timeout, log_level=args.log_level,
            )
            if rc != 2:
                break
            after = _trials_recorded()
            if after > before:
                no_progress = 0
            else:
                no_progress += 1
                if no_progress >= MAX_NO_PROGRESS:
                    log.warning("%s/%s: %d retries with no new trials — giving up",
                                op, kn, no_progress)
                    break
            log.info("Retry for %s/%s (progress %d->%d, no_progress=%d)",
                     op, kn, before, after, no_progress)
        elapsed = time.time() - t0

        envelope = store.load_operator_results(op, kn)
        shape_results = envelope.get("shapes", []) if envelope else []
        ok = sum(1 for r in shape_results if r.get("status") == "success")
        err = sum(1 for r in shape_results if r.get("status") == "error")
        improved = sum(1 for r in shape_results
                       if r.get("speedup") and r["speedup"] > 1.005)

        if rc == 0 and ok > 0:
            status = "completed"; completed += 1
        else:
            status = "failed"; failed += 1

        store.save_operator_results(
            operator=op, kernel=kn, shape_results=shape_results,
            status=status, extra={"has_autotune": has_autotune,
                                  "harness": "compare_configs"},
        )
        print(f"  Done in {elapsed:.1f}s — shapes={len(shape_results)}, "
              f"ok={ok}, errors={err}, improved={improved}")

    print(f"\n{'='*70}\n  FINAL SUMMARY\n{'='*70}")
    print(f"  Total:      {total}")
    print(f"  Completed:  {completed}")
    print(f"  Failed:     {failed}")
    print(f"  Skipped:    {skipped}")
    store.print_summary()
    if args.export_csv:
        csv_path = store.export_csv()
        print(f"  CSV:        {csv_path}")
    summary_path = store.device_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(store.export_summary(), f, indent=2, default=str)
    print(f"  Summary:    {summary_path}")


if __name__ == "__main__":
    main()
