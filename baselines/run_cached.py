#!/usr/bin/env python3
"""Replay scaled-budget strategies off the autotune latency cache.

No GPU required.  For each (operator, kernel, seed, shape):

1. Resolve search space + scaled budget.
2. Instantiate the chosen strategy class.
3. For ``budget`` trials: propose a config, look up its latency in the
   autotune cache, feed back to the strategy.
4. Write the same per-shape JSON envelope that the GPU runners produce,
   under ``baselines/results/<device>/<strategy>_cached/``.

Kernels whose autotune cache used the ``@triton.autotune`` decorator's
hand-picked config list (not the full common search space) are skipped
with a warning -- cache coverage is too sparse to replay faithfully.

Usage::

    python -m baselines.run_cached \\
        --cache-dir baselines/results/v100/autotune \\
        --strategy random_search bo auc_bandit \\
        --seeds 0 1 2 3 4 \\
        --output-dir baselines/results/v100

The output layout mirrors the GPU runs so ``analyze_comparison.py``
can ingest them without modification.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.common.budget import scaled_budget, space_size
from baselines.common.latency_cache import AutotuneLatencyCache
from baselines.common.search_space import get_search_space

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

def _make_strategy(name: str, search_space: Dict[str, Any],
                   n_trials: int, seed: int):
    if name == "random_search":
        from baselines.random_search.random_search import RandomSearch
        return RandomSearch(search_space, n_trials=n_trials, seed=seed)
    if name == "bo":
        from baselines.bo.bo_search import BOSearch
        return BOSearch(search_space, n_trials=n_trials, seed=seed)
    if name == "auc_bandit":
        from baselines.auc_bandit.search import AUCBanditSearch
        return AUCBanditSearch(search_space, n_trials=n_trials, seed=seed)
    raise ValueError(f"Unknown strategy: {name}")


# ---------------------------------------------------------------------------
# Envelope construction -- mirrors schemas written by the GPU runners
# ---------------------------------------------------------------------------

def _shape_result(
    strategy: str,
    operator: str,
    kernel: str,
    shape_id: Any,
    x_val: Optional[str],
    triton_impl_name: Optional[str],
    search_space: Dict[str, Any],
    n_trials: int,
    budget_policy: str,
    original_lat: Optional[float],
    search,
    trial_errors: int,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    best_cfg, best_lat, best_id = search.best()
    speedup = (
        (original_lat / best_lat)
        if (original_lat is not None and best_lat is not None and best_lat > 0)
        else None
    )

    if best_cfg is None:
        status = "error"
        error_msg = f"All {n_trials} trials were cache miss/failure"
    else:
        status = "success"
        error_msg = None

    def _cfg_strkeys(c):
        if c is None:
            return None
        return {str(k): v for k, v in c.items()}

    # History may have different key names across strategies; normalize.
    history = search.history()

    result = {
        "operator": operator,
        "kernel": kernel,
        "input_id": shape_id,
        "x_val": x_val,
        "has_autotune": False,
        "triton_impl_name": triton_impl_name,
        "n_trials": n_trials,
        "budget": n_trials,
        "budget_policy": budget_policy,
        "space_size": space_size(search_space),
        "original_config_latency_ms": original_lat,
        "best_latency_ms": best_lat,
        "latency_ms": best_lat,
        "speedup": speedup,
        "trial_errors": trial_errors,
        "status": status,
        "error": error_msg,
        "verified": status == "success",
        "verification_error": None,
    }

    # Strategy-specific fields so the analyzer / result stores find the
    # history under their expected keys.
    if strategy == "bo":
        result["bo_config"] = _cfg_strkeys(best_cfg)
        result["bo_best_trial_id"] = best_id
        result["bo_trial_history"] = history
        result["bo_trial_errors"] = trial_errors
    else:
        result["best_config"] = _cfg_strkeys(best_cfg)
        result["best_trial_id"] = best_id
        result["trial_history"] = history

    if strategy == "auc_bandit":
        result["window"] = search.window
        result["de_pop_size"] = search.de_pop_size
        result["technique_trials"] = search.technique_trials()
        if best_id is not None and 0 <= best_id < len(history):
            result["best_proposer"] = history[best_id].get("proposer")
        else:
            result["best_proposer"] = None

    result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Per-(kernel, seed) replay
# ---------------------------------------------------------------------------

def replay_kernel(
    strategy_name: str,
    cache: AutotuneLatencyCache,
    operator: str,
    kernel: str,
    seed: int,
    budget_policy: str = "scaled",
    fixed_n_trials: int = 9,
    missing_as_failure: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Run the strategy across every shape in the cache for this kernel.

    Returns (shape_results_list, per-kernel stats dict).
    """
    search_space = get_search_space(operator, kernel)
    if budget_policy == "scaled":
        n_trials = scaled_budget(space_size(search_space))
    else:
        n_trials = int(fixed_n_trials)

    results: List[Dict[str, Any]] = []
    stats = {"shape_ok": 0, "shape_err": 0,
             "trials_total": 0, "trials_success": 0,
             "trials_failure": 0, "trials_miss": 0}

    for shape_id in cache.shape_ids(operator, kernel):
        sc = cache.shape(operator, kernel, shape_id)
        search = _make_strategy(
            strategy_name, search_space, n_trials=n_trials, seed=seed,
        )
        trial_errors = 0
        per_trial_miss = 0
        for _ in range(n_trials):
            cfg = search.propose()
            status, lat = cache.lookup(operator, kernel, shape_id, cfg)
            stats["trials_total"] += 1
            if status == "success":
                stats["trials_success"] += 1
                search.observe(lat)
            elif status == "failure":
                stats["trials_failure"] += 1
                trial_errors += 1
                search.observe(None)
            else:  # miss
                stats["trials_miss"] += 1
                per_trial_miss += 1
                if missing_as_failure:
                    trial_errors += 1
                    search.observe(None)
                else:
                    # Allow caller to handle misses differently, but we
                    # still need to resolve the pending propose() so
                    # next iteration can call propose() again.
                    search.observe(None)
                    trial_errors += 1

        row = _shape_result(
            strategy=strategy_name,
            operator=operator,
            kernel=kernel,
            shape_id=shape_id,
            x_val=sc.x_val if sc else None,
            triton_impl_name=sc.triton_impl_name if sc else None,
            search_space=search_space,
            n_trials=n_trials,
            budget_policy=budget_policy,
            original_lat=sc.original_latency_ms if sc else None,
            search=search,
            trial_errors=trial_errors,
            extra={
                "seed": seed,
                "cache_source": "autotune",
                "cache_misses": per_trial_miss,
            },
        )
        results.append(row)
        if row["status"] == "success":
            stats["shape_ok"] += 1
        else:
            stats["shape_err"] += 1

    return results, stats


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------

def _device_from_cache_dir(cache_dir: Path) -> str:
    # cache_dir is .../<device>/autotune -- return <device>
    return cache_dir.parent.name


def _strategy_output_dir(
    output_dir: Path, device: str, strategy: str,
) -> Path:
    d = output_dir / device / f"{strategy}_cached"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_envelope(
    out_dir: Path, operator: str, kernel: str,
    shape_results: List[Dict[str, Any]], status: str,
    strategy: str, seed: int, budget_policy: str,
) -> Path:
    payload = {
        "operator": operator,
        "kernel": kernel,
        "device": out_dir.parent.name,
        "baseline": f"{strategy}_cached",
        "num_shapes": len(shape_results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "seed": seed,
        "budget_policy": budget_policy,
        "cache_source": "autotune",
        "shapes": shape_results,
    }
    path = out_dir / f"{operator}__{kernel}.json"
    # If we already wrote this kernel for an earlier seed, append the seed
    # to the filename so multi-seed runs don't overwrite each other.
    if path.exists():
        path = out_dir / f"{operator}__{kernel}__seed{seed}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Replay scaled-budget strategies off the autotune cache (no GPU required)",
    )
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Autotune results dir, e.g. baselines/results/v100/autotune")
    parser.add_argument("--strategy", nargs="+", required=True,
                        choices=["random_search", "bo", "auc_bandit"],
                        help="One or more strategies to replay")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--run-list", type=Path, default=None,
                        help="Optional run-list JSON to filter kernels")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Results root (default: baselines/results)")
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--budget-policy", default="scaled",
                        choices=["fixed", "scaled"])
    parser.add_argument("--n-trials", type=int, default=9,
                        help="Per-shape trial count when --budget-policy=fixed")
    parser.add_argument(
        "--include-decorator-kernels", action="store_true", default=False,
        help="Replay even kernels whose autotune cache used "
             "@triton.autotune's config list (not recommended; cache "
             "is sparse and strategies will mostly see misses)",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cache_dir = args.cache_dir.resolve()
    device = _device_from_cache_dir(cache_dir)
    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else _PROJECT_ROOT / "baselines" / "results"
    )

    print(f"\nCache-replay runner")
    print(f"  cache-dir:   {cache_dir}")
    print(f"  device:      {device}")
    print(f"  output-dir:  {output_dir}")
    print(f"  strategies:  {args.strategy}")
    print(f"  seeds:       {args.seeds}")
    print(f"  budget:      {args.budget_policy}"
          + (f" (n_trials={args.n_trials})" if args.budget_policy == "fixed" else ""))

    cache = AutotuneLatencyCache.load(cache_dir)

    # Filter kernels
    all_kernels = cache.kernels()
    wanted: List[Tuple[str, str]] = list(all_kernels)

    if args.run_list:
        from baselines.common.discover import load_run_list
        rl = set(load_run_list(args.run_list))
        wanted = [k for k in wanted if k in rl]
    if args.operator:
        wanted = [k for k in wanted if k[0] == args.operator]
    if args.kernel:
        wanted = [k for k in wanted if k[1] == args.kernel]

    # Split into cache-eligible vs skipped
    eligible: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str, str]] = []  # (op, kern, reason)
    for op, kern in wanted:
        if cache.is_common_space_kernel(op, kern):
            eligible.append((op, kern))
        elif args.include_decorator_kernels:
            eligible.append((op, kern))
        else:
            cov = cache.coverage(op, kern)
            src = ",".join(cov["sources"].keys())
            skipped.append((op, kern, f"config_source={src} -- cache only covers {cov['n_configs_expected']} configs"))

    print(f"\n  kernels in cache:   {len(all_kernels)}")
    print(f"  kernels matching:   {len(wanted)}")
    print(f"  eligible (replay):  {len(eligible)}")
    print(f"  skipped:            {len(skipped)}")
    if skipped:
        print("  skipped kernels:")
        for op, kern, reason in skipped:
            print(f"    - {op}/{kern[:55]} -- {reason}")

    if not eligible:
        print("\nNothing to replay.")
        return

    # ---- Replay ----
    total_runs = len(eligible) * len(args.strategy) * len(args.seeds)
    done = 0
    t0 = time.time()
    agg_stats = {"shape_ok": 0, "shape_err": 0, "trials_total": 0,
                 "trials_success": 0, "trials_failure": 0, "trials_miss": 0}

    for strategy in args.strategy:
        out_dir = _strategy_output_dir(output_dir, device, strategy)
        for seed in args.seeds:
            for op, kern in eligible:
                done += 1
                t_start = time.time()
                results, stats = replay_kernel(
                    strategy_name=strategy,
                    cache=cache,
                    operator=op,
                    kernel=kern,
                    seed=seed,
                    budget_policy=args.budget_policy,
                    fixed_n_trials=args.n_trials,
                )
                ok = stats["shape_ok"]
                err = stats["shape_err"]
                env_path = _save_envelope(
                    out_dir, op, kern, results,
                    "completed" if ok > 0 else "failed",
                    strategy, seed, args.budget_policy,
                )
                for k in agg_stats:
                    agg_stats[k] += stats[k]
                elapsed = time.time() - t_start
                print(f"  [{done}/{total_runs}] {strategy}/seed{seed} "
                      f"{op}/{kern[:40]}: "
                      f"shapes={ok}+{err} "
                      f"trials={stats['trials_total']} "
                      f"({stats['trials_success']}s/{stats['trials_failure']}f/"
                      f"{stats['trials_miss']}m) "
                      f"{elapsed:.2f}s -> {env_path.name}")

    total_elapsed = time.time() - t0
    print(f"\n{'=' * 78}")
    print(f"  Cache-replay complete in {total_elapsed:.1f}s")
    print(f"  Total runs:        {total_runs}")
    print(f"  Shapes ok/err:     {agg_stats['shape_ok']} / {agg_stats['shape_err']}")
    print(f"  Trials total:      {agg_stats['trials_total']}")
    print(f"    cache success:   {agg_stats['trials_success']} "
          f"({agg_stats['trials_success'] * 100 / max(agg_stats['trials_total'], 1):.1f}%)")
    print(f"    cache failure:   {agg_stats['trials_failure']}")
    print(f"    cache miss:      {agg_stats['trials_miss']}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
