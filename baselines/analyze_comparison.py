#!/usr/bin/env python3
"""Unified analysis for the scaled-budget baselines comparison.

Consumes per-kernel envelopes from
``baselines/results/<device>/{random,bo,bo_scaled,random_search,auc_bandit,autotune}/*.json``
plus (optionally) LLM multi-agent per-shape outputs, and produces:

* Per-kernel convergence plots (matched@9 and full-budget views).
* Per-family aggregate convergence plots using normalized speedup.
* Cross-kernel heatmap of normalized speedup.
* ``summary.json`` with a nested ``{kernel: {strategy: {iter_9: {...},
  full: {...}}}}`` dict, and an ``aggregate`` block with per-kernel-then-
  cross-kernel GeoMean speedup (see memory note: geomean must be
  per-kernel-first, then geomean across kernels).

The script is strategy-agnostic: any baseline JSON that carries a
``shapes[].bo_trial_history`` OR ``shapes[].trial_history`` list is
unpacked into a per-trial convergence series.  Baselines that store only
a final latency (existing ``random``, ``autotune``) contribute a single
"budget=1" data point and show up in the boxplots but not the convergence
curves.

Usage::

    python baselines/analyze_comparison.py \\
        baselines/results/v100 \\
        --llm-dir multiagent_runs/v100 \\
        --output-dir baselines/results/v100/comparison
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# NOTE: we deliberately do NOT import baselines.common.search_space here
# because that module imports ``triton``, which during its own import
# pulls in ``tempfile`` -> ``random``.  Running this script directly as
# ``python baselines/analyze_comparison.py`` puts ``baselines/`` on
# ``sys.path[0]`` and the stdlib ``random`` gets shadowed by our
# ``baselines.random`` package.  Invoke with ``python -m
# baselines.analyze_comparison`` to be safe, but also keep this
# analyzer triton-free so it works either way.

log = logging.getLogger("analyze_comparison")


# ---------------------------------------------------------------------------
# Families (keyed by search-space size -- matches prompt's groupings)
# ---------------------------------------------------------------------------

FAMILY_BY_SIZE = {
    16:   "minimal_16",
    72:   "pointwise_72",
    80:   "pointwise_80",
    84:   "pointwise_84",
    96:   "pointwise_96",
    288:  "reduction_288",
    420:  "reduction_420",
    576:  "gemm_576",
    2304: "gemm_2304",
}


# ---------------------------------------------------------------------------
# Envelope normalization -- one row per (strategy, kernel, shape, seed, trial)
# ---------------------------------------------------------------------------

def _history_from_shape(shape: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the per-trial history list from either schema variant."""
    if "bo_trial_history" in shape and isinstance(shape["bo_trial_history"], list):
        return shape["bo_trial_history"]
    if "trial_history" in shape and isinstance(shape["trial_history"], list):
        return shape["trial_history"]
    return []


def _best_so_far_series(history: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Running min over valid latencies; ``None`` until the first success."""
    series: List[Optional[float]] = []
    best: Optional[float] = None
    for entry in history:
        lat = entry.get("latency_ms")
        if (lat is not None and lat == lat and lat > 0
                and entry.get("state", "COMPLETE") == "COMPLETE"):
            if best is None or lat < best:
                best = float(lat)
        series.append(best)
    return series


def _kernel_key(op: str, kernel: Optional[str]) -> str:
    return f"{op}__{kernel}" if kernel else op


def load_baseline_dir(
    baseline_dir: Path, strategy_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parse every ``*__*.json`` envelope under ``baseline_dir``."""
    label = strategy_label or baseline_dir.name
    rows: List[Dict[str, Any]] = []

    for path in sorted(baseline_dir.glob("*.json")):
        if path.name in ("index.json", "summary.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            log.warning("failed to parse %s: %s", path, exc)
            continue
        op = data.get("operator", path.stem)
        kernel = data.get("kernel")
        seed = data.get("seed") or data.get("random_seed") or 0
        if "extra" in data and "seed" in data.get("extra", {}):
            seed = data["extra"]["seed"]
        for shape in data.get("shapes", []):
            history = _history_from_shape(shape)
            final_lat = shape.get("best_latency_ms") or shape.get("latency_ms")
            orig_lat = shape.get("original_config_latency_ms")
            speedup = shape.get("speedup")
            budget = shape.get("budget") or shape.get("n_trials") or (
                len(history) if history else 1
            )
            if history:
                bsf = _best_so_far_series(history)
            else:
                # Baselines without trial history (existing random/autotune):
                # use the single final latency as a one-element series.
                bsf = [final_lat] if final_lat else []
            rows.append({
                "strategy": label,
                "operator": op,
                "kernel": kernel,
                "kernel_key": _kernel_key(op, kernel),
                "shape_id": shape.get("input_id"),
                "x_val": shape.get("x_val"),
                "seed": int(seed) if isinstance(seed, (int, str)) else 0,
                "budget": int(budget),
                "final_latency_ms": float(final_lat) if final_lat else None,
                "original_latency_ms": float(orig_lat) if orig_lat else None,
                "speedup_vs_original": float(speedup) if speedup else None,
                "best_so_far_series": bsf,
                "status": shape.get("status"),
            })
    return rows


def load_llm_dir(llm_dir: Path) -> List[Dict[str, Any]]:
    """Parse LLM multi-agent per-shape outputs if provided.

    Expected schema (from ``multiagent_triton_runner.py``): each file is a
    JSON with ``kernel``, ``performance_comparison.baseline_time_ms``,
    ``performance_comparison.optimized_time_ms``, ``benchmark.speedup``.
    """
    rows: List[Dict[str, Any]] = []
    if not llm_dir.is_dir():
        return rows
    for path in sorted(llm_dir.glob("*.json")):
        if path.name in ("summary.json", "live_summary.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        kernel_full = data.get("kernel") or path.stem
        # Usually the format is "operator__kernel" already; pass through.
        op = kernel_full.split("__", 1)[0] if "__" in kernel_full else kernel_full
        kern = kernel_full.split("__", 1)[1] if "__" in kernel_full else None
        perf = data.get("performance_comparison") or {}
        bench = data.get("benchmark") or {}
        best_lat = (
            perf.get("optimized_time_ms")
            or (bench.get("recommended_stats") or {}).get("median")
        )
        orig_lat = (
            perf.get("baseline_time_ms")
            or (bench.get("original_stats") or {}).get("median")
        )
        speedup = (
            perf.get("speedup") or bench.get("speedup")
        )
        rows.append({
            "strategy": "llm_multi_agent",
            "operator": op,
            "kernel": kern,
            "kernel_key": _kernel_key(op, kern),
            "shape_id": None,
            "x_val": None,
            "seed": 0,
            "budget": 9,  # documented LLM budget
            "final_latency_ms": float(best_lat) if best_lat else None,
            "original_latency_ms": float(orig_lat) if orig_lat else None,
            "speedup_vs_original": float(speedup) if speedup else None,
            # LLM system doesn't record per-trial traces; treat as a
            # single final latency at iter 9.
            "best_so_far_series": [best_lat] * 9 if best_lat else [],
            "status": "success" if best_lat else "error",
        })
    return rows


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _value_at(series: List[Optional[float]], iter_idx: int) -> Optional[float]:
    """Return best_so_far at iter_idx (0-indexed); clamp to last available."""
    if not series:
        return None
    if iter_idx < 0:
        return None
    if iter_idx >= len(series):
        return series[-1]
    return series[iter_idx]


def _truncate_at(series: List[Optional[float]], budget: int) -> List[Optional[float]]:
    return series[:budget] if series else series


def per_run_stats(
    rows: List[Dict[str, Any]], at_iter: int,
) -> Dict[Tuple[str, str, str], List[float]]:
    """Group final latency @ iter by (strategy, kernel_key, shape_id).

    Returns dict mapping to list of latencies across seeds.
    """
    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        lat = _value_at(r["best_so_far_series"], at_iter)
        if lat is None:
            continue
        key = (r["strategy"], r["kernel_key"], str(r["shape_id"]))
        buckets[key].append(float(lat))
    return buckets


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _geomean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce the nested {kernel: {strategy: {iter_9, full}}} summary.

    Also computes cross-kernel GeoMean speedups using the memory-noted
    convention: per-kernel first, then geomean across kernels.
    """
    # Group by strategy + kernel_key
    by_sk: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sk[(r["strategy"], r["kernel_key"])].append(r)

    kernel_keys = sorted({r["kernel_key"] for r in rows})
    strategies = sorted({r["strategy"] for r in rows})

    # Find best-overall latency per (kernel, shape) across all strategies
    # at full budget.  Used to compute normalized speedup.
    best_overall: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if not r["best_so_far_series"]:
            continue
        final = r["best_so_far_series"][-1]
        if final is None:
            continue
        key = (r["kernel_key"], str(r["shape_id"]))
        if key not in best_overall or final < best_overall[key]:
            best_overall[key] = float(final)

    out: Dict[str, Any] = {
        "kernels": {},
        "strategies": strategies,
        "kernel_keys": kernel_keys,
    }

    for kk in kernel_keys:
        kernel_block: Dict[str, Any] = {}
        for strategy in strategies:
            runs = by_sk.get((strategy, kk), [])
            if not runs:
                continue

            # iter=9 matched-budget view (index 8 = 9th iter, 0-indexed)
            at9 = [
                _value_at(r["best_so_far_series"], 8)
                for r in runs
            ]
            at9 = [x for x in at9 if x is not None]

            # full-budget view
            full = [
                r["best_so_far_series"][-1]
                for r in runs if r["best_so_far_series"]
                and r["best_so_far_series"][-1] is not None
            ]

            # Per-shape normalized speedup = best_overall_for_shape / this_run
            norm_at9 = []
            norm_full = []
            for r in runs:
                shape_key = (r["kernel_key"], str(r["shape_id"]))
                overall = best_overall.get(shape_key)
                if overall is None:
                    continue
                v9 = _value_at(r["best_so_far_series"], 8)
                if v9 and v9 > 0:
                    norm_at9.append(overall / v9)
                if r["best_so_far_series"]:
                    vf = r["best_so_far_series"][-1]
                    if vf and vf > 0:
                        norm_full.append(overall / vf)

            kernel_block[strategy] = {
                "n_runs": len(runs),
                "iter_9": {
                    "mean_ms": (sum(at9) / len(at9)) if at9 else None,
                    "min_ms": min(at9) if at9 else None,
                    "max_ms": max(at9) if at9 else None,
                    "geomean_norm_speedup": _geomean(norm_at9),
                    "n_observed": len(at9),
                },
                "full": {
                    "mean_ms": (sum(full) / len(full)) if full else None,
                    "min_ms": min(full) if full else None,
                    "max_ms": max(full) if full else None,
                    "geomean_norm_speedup": _geomean(norm_full),
                    "n_observed": len(full),
                },
            }
        if kernel_block:
            out["kernels"][kk] = kernel_block

    # Cross-kernel GeoMean (per-kernel-first-then-geomean -- see memory note)
    aggregate: Dict[str, Dict[str, Optional[float]]] = {}
    for strategy in strategies:
        per_kernel_at9 = []
        per_kernel_full = []
        for kk, kern_block in out["kernels"].items():
            sb = kern_block.get(strategy)
            if not sb:
                continue
            if sb["iter_9"]["geomean_norm_speedup"] is not None:
                per_kernel_at9.append(sb["iter_9"]["geomean_norm_speedup"])
            if sb["full"]["geomean_norm_speedup"] is not None:
                per_kernel_full.append(sb["full"]["geomean_norm_speedup"])
        aggregate[strategy] = {
            "cross_kernel_geomean_at_iter_9": _geomean(per_kernel_at9),
            "cross_kernel_geomean_at_full_budget": _geomean(per_kernel_full),
            "kernels_represented": len(per_kernel_at9),
        }
    out["aggregate"] = aggregate

    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_kernel_convergence(
    rows: List[Dict[str, Any]], kernel_key: str,
    output_path: Path, matched_at_9: bool = False,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plot %s", output_path)
        return

    by_strategy: Dict[str, List[List[float]]] = defaultdict(list)
    max_len = 0
    for r in rows:
        if r["kernel_key"] != kernel_key:
            continue
        s = r["best_so_far_series"]
        if matched_at_9:
            s = _truncate_at(s, 9)
        valid = [x for x in s if x is not None]
        if not valid:
            continue
        by_strategy[r["strategy"]].append(s)
        max_len = max(max_len, len(s))

    if not by_strategy:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, series_list in sorted(by_strategy.items()):
        # Pad each series to max_len with last observed value
        padded = []
        for s in series_list:
            filled = []
            last = None
            for v in s:
                if v is not None:
                    last = v
                filled.append(last if last is not None else float("nan"))
            if len(filled) < max_len:
                filled.extend([filled[-1] if filled else float("nan")] * (max_len - len(filled)))
            padded.append(filled[:max_len])
        import numpy as np
        arr = np.array(padded, dtype=float)
        with_data = ~np.isnan(arr)
        means = np.where(
            with_data.any(axis=0),
            np.nanmean(arr, axis=0),
            np.nan,
        )
        x = list(range(1, max_len + 1))
        ax.plot(x, means, label=f"{strategy} (n={len(series_list)})", linewidth=1.5)

    ax.set_xlabel("Trial")
    ax.set_ylabel("best_so_far latency (ms) -- lower is better")
    title = f"{kernel_key} -- matched@9" if matched_at_9 else f"{kernel_key} -- full budget"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_heatmap(
    summary: Dict[str, Any], output_path: Path, which: str = "iter_9",
) -> None:
    """Heatmap of cross-kernel normalized geomean speedup per strategy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib not available; skipping heatmap")
        return

    kernels = list(summary.get("kernels", {}).keys())
    strategies = summary.get("strategies", [])
    if not kernels or not strategies:
        return

    data = np.full((len(kernels), len(strategies)), np.nan)
    for i, kk in enumerate(kernels):
        for j, s in enumerate(strategies):
            blk = summary["kernels"][kk].get(s)
            if blk and blk[which]["geomean_norm_speedup"] is not None:
                data[i, j] = blk[which]["geomean_norm_speedup"]

    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 1.3),
                                     max(4, len(kernels) * 0.35)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels, fontsize=7)
    ax.set_title(f"Normalized speedup ({'matched@9' if which == 'iter_9' else 'full budget'}) -- 1.0 = best across methods")
    for i in range(len(kernels)):
        for j in range(len(strategies)):
            v = data[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v > 0.5 else "white", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified analysis for scaled-budget classical baselines + LLM",
    )
    parser.add_argument("results_root", type=Path,
                        help="Device directory (e.g. baselines/results/v100)")
    parser.add_argument("--llm-dir", type=Path, default=None,
                        help="Directory of LLM multi-agent per-shape JSONs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strategies", nargs="*", default=None,
                        help="Subset of strategy names (subdirs) to include")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_root}")

    output_dir = (args.output_dir or results_root / "comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all baseline subdirectories ----
    rows: List[Dict[str, Any]] = []
    for sub in sorted(results_root.iterdir()):
        if not sub.is_dir() or sub.name == "comparison":
            continue
        if args.strategies and sub.name not in args.strategies:
            continue
        log.info("Loading strategy '%s' from %s", sub.name, sub)
        rows.extend(load_baseline_dir(sub, strategy_label=sub.name))

    if args.llm_dir and args.llm_dir.is_dir():
        log.info("Loading LLM multi-agent traces from %s", args.llm_dir)
        rows.extend(load_llm_dir(args.llm_dir))

    if not rows:
        log.error("No data loaded under %s", results_root)
        sys.exit(1)

    # ---- Summary ----
    summary = compute_summary(rows)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("summary.json -> %s", summary_path)

    # Console summary
    print("\n" + "=" * 78)
    print(" Cross-kernel normalized speedup (GeoMean; 1.0 = best method per kernel)")
    print("=" * 78)
    print(f"  {'Strategy':<25} {'matched@9':>15} {'full budget':>15} {'kernels':>8}")
    print(f"  {'-' * 25} {'-' * 15} {'-' * 15} {'-' * 8}")
    for s, blk in sorted(summary["aggregate"].items()):
        at9 = blk["cross_kernel_geomean_at_iter_9"]
        full = blk["cross_kernel_geomean_at_full_budget"]
        at9_s = f"{at9:.4f}" if at9 else "--"
        full_s = f"{full:.4f}" if full else "--"
        print(f"  {s:<25} {at9_s:>15} {full_s:>15} {blk['kernels_represented']:>8}")
    print("=" * 78 + "\n")

    # ---- Per-kernel convergence plots ----
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    kernel_keys = sorted({r["kernel_key"] for r in rows})
    for kk in kernel_keys:
        safe = kk.replace("/", "_").replace(" ", "_")
        plot_per_kernel_convergence(
            rows, kk, plots_dir / f"convergence_{safe}_matched9.png",
            matched_at_9=True,
        )
        plot_per_kernel_convergence(
            rows, kk, plots_dir / f"convergence_{safe}_full.png",
            matched_at_9=False,
        )
    log.info("Wrote %d convergence plots", 2 * len(kernel_keys))

    # ---- Heatmap ----
    plot_heatmap(summary, plots_dir / "heatmap_matched9.png", which="iter_9")
    plot_heatmap(summary, plots_dir / "heatmap_full.png", which="full")
    log.info("Wrote heatmaps")

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
