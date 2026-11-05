#!/usr/bin/env python3
"""CLI entry point for the uniform-random-search baseline.

Mirrors ``baselines.bo.run_all`` but spawns
``baselines.random_search.operator_runner`` subprocesses.
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
from typing import Any, Dict, List, Optional, Set

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.common.budget import print_budget_table  # noqa: E402
from baselines.common.cc_bench import enumerate_shape_ids  # noqa: E402
from baselines.common.discover import (  # noqa: E402
    discover_kernels,
    filter_by_run_list,
    get_device_name,
    load_run_list,
    print_discovery_summary,
)
from baselines.common.logging_setup import (  # noqa: E402
    default_log_path,
    setup_run_logging,
)
from baselines.common.search_space import get_search_space  # noqa: E402
from baselines.random_search.results import RandomSearchResultsStore  # noqa: E402

_DEFAULT_RUN_LIST = (
    _PROJECT_ROOT / "run_lists" / "v100_threshold_include_no_jagged_ln.json"
)

log = logging.getLogger(__name__)


def _run_subprocess(
    operator: str,
    kernel: str,
    has_autotune: bool,
    n_trials: int,
    seed: int,
    device: str,
    warmup: int,
    rep: int,
    timeout: int,
    log_level: str,
    budget_policy: str = "scaled",
    harness: str = "compare_configs",
    ir_dir: Optional[Path] = None,
    skip_shape_ids: Optional[set] = None,
    prior_results: Optional[List[Dict[str, Any]]] = None,
    per_shape_timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix=f"rs_{operator}_{kernel}_",
        delete=False, mode="w",
    ) as tmp:
        output_path = tmp.name
        if prior_results:
            json.dump(prior_results, tmp, default=str)

    cmd = [
        sys.executable, "-m", "baselines.random_search.operator_runner",
        "--operator", operator,
        "--kernel", kernel,
        "--n-trials", str(n_trials),
        "--seed", str(seed),
        "--device", device,
        "--warmup", str(warmup),
        "--rep", str(rep),
        "--budget-policy", budget_policy,
        "--harness", harness,
        "--skip-shape-ids", ",".join(str(i) for i in (skip_shape_ids or ())),
        "--output", output_path,
        "--log-level", log_level,
    ]
    if per_shape_timeout_s is not None:
        cmd.extend(["--per-shape-timeout-s", str(per_shape_timeout_s)])
    if has_autotune:
        cmd.append("--has-autotune")
    if harness == "compare_configs" and ir_dir is not None:
        cmd.extend(["--ir-dir", str(ir_dir)])

    log.info("Spawning: %s", " ".join(cmd))

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(_PROJECT_ROOT), text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ,
                 "PYTHONPATH": str(_PROJECT_ROOT),
                 "TRITON_PRINT_AUTOTUNING": "0",
                 "PYTHONUNBUFFERED": "1"},
        )
        prefix = f"[{operator}/{kernel}]"
        deadline = time.time() + timeout
        timed_out = False
        # select() loop so the deadline check still fires when the child
        # stops producing stdout (e.g. stuck in a CUDA syscall).
        import select
        try:
            assert proc.stdout is not None
            stdout_fd = proc.stdout
            while True:
                if time.time() > deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if proc.poll() is not None:
                    for line in stdout_fd:
                        log.info("%s %s", prefix, line.rstrip())
                    break
                ready, _, _ = select.select([stdout_fd], [], [], 1.0)
                if ready:
                    line = stdout_fd.readline()
                    if not line:
                        break
                    log.info("%s %s", prefix, line.rstrip())
            proc.wait(timeout=max(1.0, min(5.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            log.error("Timeout (%ds) for %s/%s", timeout, operator, kernel)
            timed_out = True

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                with open(output_path) as f:
                    partial = json.load(f)
                if timed_out:
                    log.info("Salvaged %d shape rows from timed-out subprocess",
                             len(partial))
                elif proc.returncode != 0:
                    log.info("Salvaged %d shape rows from subprocess (exit=%d)",
                             len(partial), proc.returncode)
                return partial
            except Exception as e:
                log.warning("Failed to parse partial output %s: %s",
                            output_path, e)

        if timed_out:
            return [_err(operator, kernel, n_trials,
                         f"Subprocess timed out after {timeout}s")]
        if proc.returncode != 0:
            log.warning("Subprocess for %s/%s exited with code %d",
                        operator, kernel, proc.returncode)
        return [_err(operator, kernel, n_trials,
                     f"No output file (exit code {proc.returncode})")]

    except Exception as exc:
        log.error("Subprocess error for %s/%s: %s", operator, kernel, exc)
        return [_err(operator, kernel, n_trials, str(exc))]
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _err(operator, kernel, n_trials, error):
    return {
        "operator": operator, "kernel": kernel,
        "input_id": -1, "x_val": "N/A",
        "has_autotune": False, "triton_impl_name": None,
        "n_trials": n_trials, "budget": n_trials,
        "best_config": None,
        "best_trial_id": None, "trial_history": [], "trial_errors": n_trials,
        "original_config_latency_ms": None, "original_stats": None,
        "best_latency_ms": None, "latency_ms": None, "speedup": None,
        "status": "error", "error": error,
        "verified": False, "verification_error": error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run scaled-budget uniform-random-search baseline",
    )
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--run-list", type=Path, default=_DEFAULT_RUN_LIST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=9,
                        help="Trials per shape when --budget-policy=fixed "
                             "(default: 9, matches LLM budget).")
    parser.add_argument("--budget-policy", default="scaled",
                        choices=["fixed", "scaled"],
                        help="scaled (default): per-kernel budget = "
                             "clip(round(0.10 * |space|), 9, 150). "
                             "fixed: use --n-trials.")
    parser.add_argument("--harness", default="compare_configs",
                        choices=["tritonbench", "compare_configs"],
                        help="compare_configs (default): match paper Full-system "
                             "path. tritonbench: legacy op.get_example_inputs "
                             "path (not comparable to Full).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout per subprocess. With "
                             "--per-shape-isolation this becomes the "
                             "per-shape timeout.")
    parser.add_argument("--per-shape-isolation", action="store_true",
                        default=False,
                        help="Spawn one subprocess per (kernel, shape) so "
                             "every shape gets a fresh CUDA context. "
                             "Prevents one bad config from poisoning "
                             "later shapes via cuda-cascade.")
    parser.add_argument("--export-csv", action="store_true", default=False)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Durable log path. Defaults to "
                             "<output-dir>/<device>/<baseline>/run.log")
    args = parser.parse_args()

    ir_dir = args.ir_dir.resolve()
    device = get_device_name(ir_dir)
    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else _PROJECT_ROOT / "baselines" / "results"
    )
    baseline_name = (
        "random_search" if args.budget_policy == "scaled" else "random_search_fixed"
    )
    log_path = args.log_file or default_log_path(
        output_dir, device, baseline_name,
    )
    setup_run_logging(level=args.log_level, log_file=log_path)
    log.info("Log file: %s", log_path)

    print(f"\nRandom-Search Baseline Runner")
    print(f"  IR directory:   {ir_dir}")
    print(f"  Device:         {device}")
    print(f"  Output:         {output_dir}")
    print(f"  Run-list:       {args.run_list}")
    print(f"  Harness:        {args.harness}")
    print(f"  Budget policy:  {args.budget_policy}")
    if args.budget_policy == "fixed":
        print(f"  n-trials:       {args.n_trials}")
    else:
        print(f"  n-trials:       (per-kernel, scaled)")
    print(f"  Seed:           {args.seed}")
    print(f"  Warmup/Rep:     {args.warmup}/{args.rep}")
    print(f"  Timeout:        {args.timeout}s per kernel")

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

    if args.budget_policy == "scaled":
        print_budget_table(entries)

    store = RandomSearchResultsStore(
        output_dir, device,
        n_trials=args.n_trials, seed=args.seed,
        baseline_name=baseline_name,
    )

    completed, failed, skipped = 0, 0, 0
    total = len(entries)

    for i, e in enumerate(entries, 1):
        op_name = e["operator"]; kernel = e["kernel"]
        has_autotune = e.get("has_autotune", False)
        num_shapes = e.get("num_shapes", "?")

        if args.resume and store.is_completed(op_name, kernel):
            print(f"\n[{i}/{total}] {op_name}/{kernel}: SKIPPED (already completed)")
            skipped += 1
            continue

        tag = "autotune" if has_autotune else "no-autotune"
        print(f"\n[{i}/{total}] {op_name}/{kernel} "
              f"({num_shapes} shapes, {tag}, policy={args.budget_policy})")

        t0 = time.time()
        # Shape-level retry mirrors baselines/bo/run_all.py: a subprocess
        # can die three ways during a kernel:
        #   1) CUDA poisoned mid-kernel — remaining shapes get
        #      "skipped: CUDA" markers in the output.
        #   2) SIGABRT / OOM / timeout — remaining shapes are missing.
        #   3) Cuda-cascade: every trial of a shape was a CUDA-related
        #      failure (root + cascaded skips).  cc_bench's per-shape
        #      cfg_cache short-circuits the poison config on the next
        #      attempt, so the remaining trials run with a fresh CUDA
        #      context.  Allow a multi-attempt recovery sequence with
        #      NO_PROGRESS_TOLERANCE headroom before bailing.
        MAX_RETRIES = 80
        expected_ids: Set[int] = (
            enumerate_shape_ids(ir_dir, op_name, kernel)
            if args.harness == "compare_configs" else set()
        )

        def _all_trials_cuda_cascade(r):
            hist = r.get("trial_history") or []
            if not hist:
                return False
            saw_cuda_related = False
            for h in hist:
                if h.get("latency_ms") is not None:
                    return False
                e = (h.get("error") or "").lower()
                if ("cuda error" in e
                        or "illegal memory access" in e
                        or "an illegal instruction" in e
                        or "context poisoned" in e):
                    saw_cuda_related = True
            return saw_cuda_related

        def _is_stable(r):
            s = r.get("status")
            if s == "in_progress":
                return False
            err = (r.get("error") or "")
            if err.startswith("skipped: CUDA"):
                return False
            if (s == "error"
                    and not r.get("recovery_attempted")
                    and _all_trials_cuda_cascade(r)):
                return False
            return s == "success" or s == "error"

        by_input: Dict[Any, Dict[str, Any]] = {}
        prev_stable_count = 0
        no_progress_streak = 0
        NO_PROGRESS_TOLERANCE = 5

        # Per-shape isolation: spawn one subprocess per (kernel, shape) so
        # every shape gets a fresh CUDA context.  A bad config on shape X
        # can no longer cascade-poison shapes X+1..N within the same
        # process.  Each subprocess hits compare_configs with
        # skip_shape_ids = expected - {target}, so it only processes the
        # one shape it was launched for.  Per-shape timeout becomes
        # args.timeout (typically much shorter than the kernel-level
        # timeout — e.g. 600s instead of 7200s).
        if args.per_shape_isolation and expected_ids:
            shape_targets = sorted(expected_ids)
            PER_SHAPE_MAX_ATTEMPTS = 3
            for sid_target in shape_targets:
                for ps_attempt in range(PER_SHAPE_MAX_ATTEMPTS):
                    cur = by_input.get(sid_target)
                    if cur is not None and _is_stable(cur):
                        break
                    skip = set(expected_ids) - {sid_target}
                    # Rotate the seed across retries so a SIGABRT-causing
                    # config (which random_search would draw at the same
                    # trial position with a fixed seed) gets pushed past.
                    # cc_bench replays the saved trial_history first, so
                    # already-tried configs are still observed; the new
                    # seed only changes which configs are proposed beyond
                    # the saved replay tail.  Reproducibility within a
                    # crash-free shape is unchanged because attempt 0
                    # always uses args.seed.
                    attempt_seed = args.seed + 1009 * ps_attempt
                    partial = _run_subprocess(
                        operator=op_name, kernel=kernel,
                        has_autotune=has_autotune,
                        n_trials=args.n_trials, seed=attempt_seed,
                        device="cuda", warmup=args.warmup, rep=args.rep,
                        timeout=args.timeout, log_level=args.log_level,
                        budget_policy=args.budget_policy,
                        harness=args.harness, ir_dir=ir_dir,
                        skip_shape_ids=skip,
                        prior_results=list(by_input.values()),
                    )
                    for r in partial:
                        sid = r.get("input_id")
                        if sid is None:
                            continue
                        prev = by_input.get(sid)
                        if (prev is None
                                or (prev.get("status") != "success"
                                    and r.get("status") == "success")):
                            by_input[sid] = r
                # Persist after every shape so a kill mid-kernel doesn't
                # lose accumulated progress.
                inter = list(by_input.values())
                inter_ok = sum(1 for r in inter if r.get("status") == "success")
                store.save_operator_results(
                    operator=op_name, kernel=kernel,
                    shape_results=inter,
                    status="completed" if inter_ok > 0 else "failed",
                    extra={
                        "has_autotune": has_autotune,
                        "n_trials": args.n_trials,
                        "budget_policy": args.budget_policy,
                    },
                )
            results = list(by_input.values())
            elapsed = time.time() - t0
            ok = sum(1 for r in results if r.get("status") == "success")
            err = sum(1 for r in results if r.get("status") == "error")
            improved = sum(1 for r in results
                           if r.get("speedup") and r["speedup"] > 1.005)
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
                extra={
                    "has_autotune": has_autotune,
                    "n_trials": args.n_trials,
                    "budget_policy": args.budget_policy,
                },
            )
            print(f"  Done in {elapsed:.1f}s — shapes={len(results)}, "
                  f"ok={ok}, errors={err}, improved={improved}")
            continue  # next kernel

        for attempt in range(MAX_RETRIES):
            skip_ids = {sid for sid, r in by_input.items() if _is_stable(r)}
            partial = _run_subprocess(
                operator=op_name, kernel=kernel, has_autotune=has_autotune,
                n_trials=args.n_trials, seed=args.seed,
                device="cuda", warmup=args.warmup, rep=args.rep,
                timeout=args.timeout, log_level=args.log_level,
                budget_policy=args.budget_policy,
                harness=args.harness, ir_dir=ir_dir,
                skip_shape_ids=skip_ids,
                prior_results=list(by_input.values()),
            )
            for r in partial:
                sid = r.get("input_id")
                if sid is None:
                    continue
                prev = by_input.get(sid)
                if (prev is None
                        or (prev.get("status") != "success"
                            and r.get("status") == "success")):
                    by_input[sid] = r
            stable_ids = {sid for sid, r in by_input.items() if _is_stable(r)}
            poisoned = {sid for sid, r in by_input.items()
                        if (r.get("error") or "").startswith("skipped: CUDA")}
            in_progress = {sid for sid, r in by_input.items()
                           if r.get("status") == "in_progress"}
            missing = (expected_ids - stable_ids - poisoned - in_progress
                       if expected_ids else set())

            intermediate_results = list(by_input.values())
            intermediate_ok = sum(
                1 for r in intermediate_results if r.get("status") == "success"
            )
            store.save_operator_results(
                operator=op_name, kernel=kernel,
                shape_results=intermediate_results,
                status="completed" if intermediate_ok > 0 else "failed",
                extra={
                    "has_autotune": has_autotune,
                    "n_trials": args.n_trials,
                    "budget_policy": args.budget_policy,
                },
            )

            if not poisoned and not missing and not in_progress:
                break

            recovery_eligible = {
                sid for sid, r in by_input.items()
                if (not _is_stable(r)
                    and not r.get("recovery_attempted")
                    and _all_trials_cuda_cascade(r))
            }

            if attempt > 0 and len(stable_ids) == prev_stable_count:
                no_progress_streak += 1
                if no_progress_streak >= NO_PROGRESS_TOLERANCE:
                    log.info(
                        "%s/%s attempt %d: no new stable shapes for %d "
                        "attempts (%d stable, %d recovery-eligible, "
                        "%d poisoned, %d missing) — bailing out",
                        op_name, kernel, attempt + 1, no_progress_streak,
                        len(stable_ids), len(recovery_eligible),
                        len(poisoned), len(missing),
                    )
                    break
            else:
                no_progress_streak = 0
            prev_stable_count = len(stable_ids)

            log.info(
                "%s/%s attempt %d: %d stable, %d poisoned, %d in-progress, "
                "%d missing — retrying",
                op_name, kernel, attempt + 1,
                len(stable_ids), len(poisoned), len(in_progress), len(missing),
            )
        results = list(by_input.values())
        elapsed = time.time() - t0

        ok = sum(1 for r in results if r.get("status") == "success")
        err = sum(1 for r in results if r.get("status") == "error")
        improved = sum(1 for r in results
                       if r.get("speedup") and r["speedup"] > 1.005)

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
            extra={
                "has_autotune": has_autotune,
                "n_trials": args.n_trials,
                "budget_policy": args.budget_policy,
            },
        )

        print(f"  Done in {elapsed:.1f}s — shapes={len(results)}, "
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
