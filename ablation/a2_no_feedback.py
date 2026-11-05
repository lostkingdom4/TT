#!/usr/bin/env python3
"""Ablation A2: No Feedback Loop — accept first-pass candidates.

Runs all kernels from v100_threshold_include.json with max_iterations=1,
disabling the benchmark-driven feedback/retry loop. Agents get one shot
to propose configs; regressions are not retried.

Question: How much does per-agent feedback contribute to final speedup?
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUN_LIST = _PROJECT_ROOT / "run_lists" / "v100_threshold_include.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "output" / "ablation_a2_no_feedback"


def main():
    parser = argparse.ArgumentParser(
        description="Ablation A2: No Feedback Loop",
    )
    parser.add_argument("--gpu", default="v100", help="GPU folder name (default: v100)")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--all-shapes", action="store_true", default=True)
    parser.add_argument("--num-shapes", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=10800,
                        help="Per-kernel timeout in seconds (default: 10800 = 3h)")
    args = parser.parse_args()

    cmd = [
        sys.executable, str(_PROJECT_ROOT / "batch_runner.py"),
        "--gpu", args.gpu,
        "--run-list", str(_RUN_LIST),
        "--output-dir", str(args.output_dir),
        "--llm-provider", args.llm_provider,
        "--max-iterations", "1",
        "--num-candidates", "3",
    ]

    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])
    if args.num_shapes is not None:
        cmd.extend(["--num-shapes", str(args.num_shapes)])
    elif args.all_shapes:
        cmd.append("--all-shapes")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.debug_mode:
        cmd.append("--debug-mode")
    if args.no_resume:
        cmd.append("--no-resume")
    cmd.extend(["--timeout", str(args.timeout)])

    print("Ablation A2: No Feedback Loop (max_iterations=1)")
    print(f"  Run list:   {_RUN_LIST}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Command:    {' '.join(cmd)}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.time() - t0

    print(f"\nAblation A2 completed in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
