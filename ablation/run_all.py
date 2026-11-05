#!/usr/bin/env python3
"""Run all ablation studies (A1a, A1b, A2, A3) sequentially.

Usage:
    # Run all ablations
    python ablation/run_all.py

    # Dry-run to see commands
    python ablation/run_all.py --dry-run

    # Run only specific ablations
    python ablation/run_all.py --ablations a1a a3

    # Pass extra flags
    python ablation/run_all.py --llm-provider openai --debug-mode
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ABLATIONS = {
    "a1a": {
        "script": "ablation/a1a_single_agent_dm_only.py",
        "name": "A1a: Single Agent (DM only)",
    },
    "a1b": {
        "script": "ablation/a1b_single_agent_rb_only.py",
        "name": "A1b: Single Agent (RB only)",
    },
    "a2": {
        "script": "ablation/a2_no_feedback.py",
        "name": "A2: No Feedback Loop",
    },
    "a3": {
        "script": "ablation/a3_single_candidate.py",
        "name": "A3: Single Candidate",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run all ablation studies sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ablations", nargs="+", default=list(ABLATIONS.keys()),
        choices=list(ABLATIONS.keys()),
        help=f"Which ablations to run (default: all). Choices: {list(ABLATIONS.keys())}",
    )
    parser.add_argument("--gpu", default="v100")
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

    selected = [a for a in ABLATIONS if a in args.ablations]

    print(f"{'='*70}")
    print(f"  Ablation Study Runner — {len(selected)} experiments")
    print(f"{'='*70}")
    for a in selected:
        print(f"  - {ABLATIONS[a]['name']}")
    print(f"{'='*70}\n")

    results = {}
    total_t0 = time.time()

    for i, ablation_id in enumerate(selected, 1):
        info = ABLATIONS[ablation_id]
        script = _PROJECT_ROOT / info["script"]

        print(f"\n{'='*70}")
        print(f"  [{i}/{len(selected)}] {info['name']}")
        print(f"{'='*70}\n")

        cmd = [sys.executable, str(script)]

        # Forward common flags
        cmd.extend(["--gpu", args.gpu])
        cmd.extend(["--llm-provider", args.llm_provider])
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

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
        elapsed = time.time() - t0

        results[ablation_id] = {
            "name": info["name"],
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
        }

        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"\n  [{ablation_id}] {status} — {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Final summary
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print("  Ablation Summary")
    print(f"{'='*70}")
    print(f"  {'ID':<6} {'Name':<35} {'Status':<10} {'Time':>8}")
    print(f"  {'-'*6} {'-'*35} {'-'*10} {'-'*8}")
    for ablation_id, r in results.items():
        status = "OK" if r["returncode"] == 0 else "FAILED"
        elapsed_str = f"{r['elapsed_s']/60:.1f}m"
        print(f"  {ablation_id:<6} {r['name']:<35} {status:<10} {elapsed_str:>8}")
    print(f"{'='*70}")
    print(f"  Total wall-clock: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"{'='*70}")

    # Exit with non-zero if any ablation failed
    if any(r["returncode"] != 0 for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
