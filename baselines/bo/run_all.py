#!/usr/bin/env python3
"""CLI entry point for the Bayesian-Optimization (Optuna TPE) baseline.

Discovers (operator, kernel) pairs from a run-list JSON (default:
``run_lists/v100_improved_only.json`` — the 19 kernels in Table 2 of
the paper) and runs the BO per-shape loop in an isolated subprocess
per kernel.

Usage examples::

    # Default: all 19 kernels, 9 trials per shape
    python -m baselines.bo.run_all \\
        --ir-dir extract/ir_output/v100 --export-csv

    # Larger BO budget
    python -m baselines.bo.run_all \\
        --ir-dir extract/ir_output/v100 --n-trials 30

    # Single kernel
    python -m baselines.bo.run_all \\
        --ir-dir extract/ir_output/v100 \\
        --operator rope --kernel triton_poi_fused_add_cat_mul_0
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

from baselines.bo.results import BOResultsStore  # noqa: E402
from baselines.common.budget import print_budget_table  # noqa: E402
from baselines.common.discover import (  # noqa: E402
    discover_kernels,
    filter_by_run_list,
    get_device_name,
    load_run_list,
    print_discovery_summary,
)
from baselines.common.cc_bench import enumerate_shape_ids  # noqa: E402
from baselines.common.logging_setup import (  # noqa: E402
    default_log_path,
    setup_run_logging,
)
from baselines.common.search_space import get_search_space  # noqa: E402

_DEFAULT_RUN_LIST = (
    _PROJECT_ROOT / "run_lists" / "v100_improved_only.json"
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
    budget_policy: str = "fixed",
    harness: str = "compare_configs",
    ir_dir: Optional[Path] = None,
    skip_shape_ids: Optional[set] = None,
    prior_results: Optional[List[Dict[str, Any]]] = None,
    per_shape_timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    # Seed the subprocess's incremental-output file with prior retries'
    # per-shape rows so cc_bench can replay partial trial history on startup.
    # Without this, each retry's fresh subprocess wouldn't see the in-progress
    # shape's saved trials and would restart that shape from trial 0.
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix=f"bo_{operator}_{kernel}_",
        delete=False, mode="w",
    ) as tmp:
        output_path = tmp.name
        if prior_results:
            json.dump(prior_results, tmp, default=str)

    cmd = [
        sys.executable, "-m", "baselines.bo.operator_runner",
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
        # Popen + line-by-line streaming so child output reaches our
        # FileHandler (and tail -f) live, rather than surfacing only
        # after the kernel finishes.
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
        # Drive a select() loop so the deadline is checked even when the
        # child stops producing stdout (e.g. stuck in a CUDA syscall).
        # The previous `for line in proc.stdout` blocked on stdout
        # forever when do_bench hung, surviving past `timeout`.
        import select
        try:
            assert proc.stdout is not None
            stdout_fd = proc.stdout
            while True:
                if time.time() > deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if proc.poll() is not None:
                    # Child already exited; drain any remaining buffered output.
                    for line in stdout_fd:
                        log.info("%s %s", prefix, line.rstrip())
                    break
                ready, _, _ = select.select([stdout_fd], [], [], 1.0)
                if ready:
                    line = stdout_fd.readline()
                    if not line:
                        break
                    log.info("%s %s", prefix, line.rstrip())
            # Child exited; reap with a short wait.
            proc.wait(timeout=max(1.0, min(5.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # SIGKILL is enough; if it survived even that, give up
            log.error("Timeout (%ds) for %s/%s", timeout, operator, kernel)
            timed_out = True

        # Always try to salvage the incremental output file first — it
        # contains per-shape rows written atomically after each shape
        # finished.  This applies on timeout, on SIGABRT, on OOM kill,
        # and on clean exit alike.  Only fall back to a synthetic error
        # stub if no partial data landed on disk.
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
        "n_trials": n_trials, "bo_config": None,
        "bo_best_trial_id": None, "bo_trial_history": [],
        "bo_trial_errors": n_trials,
        "original_config_latency_ms": None, "original_stats": None,
        "best_latency_ms": None, "latency_ms": None, "speedup": None,
        "status": "error", "error": error,
        "verified": False, "verification_error": error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the Optuna TPE BO baseline for every kernel in a run-list",
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
    parser.add_argument("--n-trials", type=int, default=9,
                        help="Optuna trials per shape when --budget-policy=fixed "
                             "(default: 9, matches LLM multi-agent budget)")
    parser.add_argument("--budget-policy", default="fixed",
                        choices=["fixed", "scaled"],
                        help="fixed: use --n-trials; scaled: per-kernel budget "
                             "= clip(round(0.10 * |search space|), 9, 150). "
                             "With 'scaled', results land in "
                             "baselines/results/<gpu>/bo_scaled/ so fixed@9 "
                             "artifacts stay intact.")
    parser.add_argument("--harness", default="compare_configs",
                        choices=["tritonbench", "compare_configs"],
                        help="compare_configs (default): match the paper's "
                             "Full-system measurement path using extracted "
                             "kernel .py + per-input shapes.json. tritonbench: "
                             "legacy op.get_example_inputs path (different "
                             "inputs; not comparable to Full).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5,
                        help="do_bench warmup (default: 5, matches paper)")
    parser.add_argument("--rep", type=int, default=20,
                        help="do_bench rep (default: 20, matches paper)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-kernel subprocess timeout in seconds")
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
    baseline_name = "bo_scaled" if args.budget_policy == "scaled" else "bo"
    log_path = args.log_file or default_log_path(
        output_dir, device, baseline_name,
    )
    setup_run_logging(level=args.log_level, log_file=log_path)
    log.info("Log file: %s", log_path)

    print(f"\nBO (Optuna TPE) Baseline Runner")
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

    print(f"  Search spaces:")
    for e in entries:
        space = get_search_space(e["operator"], e["kernel"])
        print(f"    {e['operator']}/{e['kernel']}: "
              f"{', '.join(f'{k}={v}' for k,v in space.items())}")
    print()

    if args.budget_policy == "scaled":
        print_budget_table(entries)

    store = BOResultsStore(
        output_dir, device, n_trials=args.n_trials, seed=args.seed,
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
              f"({num_shapes} shapes, {tag}, n_trials={args.n_trials})")

        t0 = time.time()
        # Shape-level retry: a subprocess can die three ways during a kernel:
        #   1) CUDA poisoned mid-kernel — remaining shapes get "skipped: CUDA"
        #      markers in the output (still present in by_input).
        #   2) SIGABRT / OOM / timeout — remaining shapes are *missing* from
        #      the output entirely (nothing in by_input for those IDs).
        #   3) All shapes processed cleanly — no retry needed.
        # We enumerate expected IDs up-front and keep retrying until every
        # expected ID has a stable (success or non-CUDA-error) entry.
        MAX_RETRIES = 80
        expected_ids: Set[int] = (
            enumerate_shape_ids(ir_dir, op_name, kernel)
            if args.harness == "compare_configs" else set()
        )
        def _is_stable(r):
            # A shape is "stable" (don't re-process) if it either succeeded
            # or errored with a non-CUDA reason.  Explicitly in_progress
            # rows (mid-shape partials) are NOT stable — the next retry
            # should resume them via cc_bench's replay logic.
            s = r.get("status")
            if s == "in_progress":
                return False
            err = (r.get("error") or "")
            if err.startswith("skipped: CUDA"):
                return False
            # CUDA-cascade-recoverable: a shape that failed every trial
            # because the first trial poisoned CUDA and the rest were
            # cascaded skips.  Allow ONE recovery retry — the per-shape
            # cfg_cache will load the poison config from the saved
            # trial_history and short-circuit it without launching the
            # GPU, so CUDA stays alive on the retry and the remaining
            # trials run normally.  The 'recovery_attempted' flag on
            # the row prevents infinite retries when the shape genuinely
            # has no recoverable configs.
            if (s == "error"
                    and not r.get("recovery_attempted")
                    and _all_trials_cuda_cascade(r)):
                return False
            return s == "success" or s == "error"

        def _all_trials_cuda_cascade(r):
            """True iff every trial in r's history is a FAIL and at
            least one of them is a CUDA-related error (root fault OR
            downstream "context poisoned" cascade).  Both patterns are
            recoverable on a fresh subprocess respawn — a real root
            fault gets cache-short-circuited, a pure cascade gets a
            clean BO retry against a fresh CUDA context."""
            hist = r.get("bo_trial_history") or r.get("trial_history") or []
            if not hist:
                return False
            saw_cuda_related = False
            for h in hist:
                if h.get("latency_ms") is not None:
                    return False  # has a successful trial; not pure cascade
                e = (h.get("error") or "").lower()
                if ("cuda error" in e
                        or "illegal memory access" in e
                        or "an illegal instruction" in e
                        or "context poisoned" in e):
                    saw_cuda_related = True
            return saw_cuda_related

        by_input: Dict[Any, Dict[str, Any]] = {}
        prev_stable_count = 0
        prev_progress_count = 0  # stable + recovery-eligible (cuda-cascade)
        no_progress_streak = 0
        # Bail only after this many consecutive attempts with NO new
        # stable shapes.  Recovery cycles (cascade -> root+cascade ->
        # cache-recovery) can take 2-3 attempts per shape, and one of
        # those attempts may not net-add a stable shape (a recovery may
        # just convert a cascade-only history into a root+cascade
        # history).  A streak of 5 wasted attempts is enough headroom
        # for that pattern while still bailing on truly stuck kernels.
        NO_PROGRESS_TOLERANCE = 5
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

            # Partial envelope save: rewrite the kernel's JSON after every
            # retry attempt so a kill / crash of the parent doesn't lose the
            # shapes we've already accumulated.  Use the same completed/failed
            # logic as the end-of-loop save.
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

            # Recovery-eligible shapes have a cuda-cascade trial_history
            # (one root CUDA fault + the rest cascaded skips).  These are
            # not stable yet but they will recover on the next attempt
            # via cc_bench's per-shape cfg_cache.  Count them as
            # "progress" so the no-progress break doesn't bail prematurely
            # on the multi-attempt cuda-recovery sequence.
            recovery_eligible = {
                sid for sid, r in by_input.items()
                if (not _is_stable(r)
                    and not r.get("recovery_attempted")
                    and _all_trials_cuda_cascade(r))
            }
            progress_count = len(stable_ids) + len(recovery_eligible)

            # Break on zero-progress: bail only after NO_PROGRESS_TOLERANCE
            # consecutive attempts with no new stable shapes.  We track
            # stable_count rather than progress_count because internal
            # recovery transitions (cascade-only -> root+cascade) keep
            # progress_count constant even when they're real progress.
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
            prev_progress_count = progress_count

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
