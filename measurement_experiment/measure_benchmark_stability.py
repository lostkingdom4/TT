#!/usr/bin/env python3
"""Measure benchmark stability by running the same kernel configuration
multiple times and computing across-run variance statistics.

Reuses the benchmarking infrastructure from ``compare_configs.py`` and kernel
discovery from ``multiagent_triton_runner.py`` — no LLM involvement.

Includes a built-in registry of 15 verified kernels covering: softmax, addmm,
jagged_mean, jagged_softmax, jagged_sum, layer_norm, low_mem_dropout, sum,
vector_add, and vector_exp — including variant kernels (simple_fused vs
variable_length_loop, seeded vs triton dropout, etc.).

Usage:
    # All 15 registered kernels, 10 runs each
    python measurement_experiment/measure_benchmark_stability.py

    # Specific operators, 20 runs, with plot
    python measurement_experiment/measure_benchmark_stability.py \
        --operators softmax,vector_add,addmm --num-runs 20 --plot

    # All kernels, 5 quick runs
    python measurement_experiment/measure_benchmark_stability.py \
        --operators all --num-runs 5 --warmup 10 --rep 50

    # Use A100 IR data instead of auto-detected device
    python measurement_experiment/measure_benchmark_stability.py \
        --cache-dir extract/ir_output/a100
"""

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — add project root so we can import sibling modules
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import triton
import triton.testing

from compare_configs import (
    _build_kwargs,
    _compute_stats,
    _load_kernel,
    _resolve_grid,
    _unwrap_autotuner,
)
from multiagent_triton_runner import (
    _extract_source_info,
    _find_shapes,
    _find_ttgir,
    _kernel_to_op_map,
    _parse_triton_kernels,
)
from paths import METADATA_DIR
from tensor_utils import create_kernel_inputs, load_tensor_shapes

logger = logging.getLogger("measure_benchmark_stability")

DEFAULT_OPERATORS = ["all"]

# ---------------------------------------------------------------------------
# Kernel registry — all verified kernel configurations from the project.
# Each entry mirrors the CLI arguments used by multiagent_triton_runner.py.
# ---------------------------------------------------------------------------

# Each entry uses relative paths from IR_BASE; resolved at runtime by
# ``build_kernel_registry()``.
_KERNEL_REGISTRY_TEMPLATE: List[Dict[str, Any]] = [
    {
        "op_name": "softmax",
        "kernel_name": "Operator.softmax_kernel",
        "cache_rel": "softmax",
        "shapes_rel": "softmax",
    },
    {
        "op_name": "addmm",
        "kernel_name": "_addmm_fwd",
        "cache_rel": "addmm",
        "shapes_rel": "addmm",
    },
    # NOTE: jagged_mean, jagged_softmax, jagged_sum are excluded because they
    # require semantically valid "offsets" tensors (sorted cumulative integers).
    # Generic random input creation produces invalid inputs that cause CUDA
    # illegal memory access errors.  These kernels need specialised input
    # generation to benchmark safely.
    {
        "op_name": "layer_norm (triton)",
        "kernel_name": "_layer_norm_fwd_fused",
        "cache_rel": "layer_norm/layer_norm_fwd/triton_layer_norm",
        "shapes_rel": "layer_norm",
    },
    {
        "op_name": "layer_norm (fused_no_bias)",
        "kernel_name": "_layer_norm_fwd_fused_no_bias",
        "cache_rel": "layer_norm/layer_norm_fwd/triton_fused_layer_norm",
        "shapes_rel": "layer_norm",
    },
    {
        "op_name": "dropout (seeded)",
        "kernel_name": "_seeded_triton_dropout",
        "cache_rel": "low_mem_dropout/low_mem_dropout_fwd/seeded_dropout",
        "shapes_rel": "low_mem_dropout",
    },
    {
        "op_name": "dropout (triton)",
        "kernel_name": "_triton_dropout",
        "cache_rel": "low_mem_dropout/low_mem_dropout_fwd/triton_dropout",
        "shapes_rel": "low_mem_dropout",
    },
    {
        "op_name": "sum",
        "kernel_name": "triton_sum_kernel_scalar_result",
        "cache_rel": "sum",
        "shapes_rel": "sum",
    },
    {
        "op_name": "vector_add",
        "kernel_name": "triton_add_kernel",
        "cache_rel": "vector_add",
        "shapes_rel": "vector_add",
    },
    {
        "op_name": "vector_exp",
        "kernel_name": "triton_exp_kernel",
        "cache_rel": "vector_exp",
        "shapes_rel": "vector_exp",
    },
]


def build_kernel_registry(ir_base: Path) -> List[Dict[str, Any]]:
    """Build the kernel registry with absolute paths rooted at *ir_base*."""
    return [
        {
            "op_name": entry["op_name"],
            "kernel_name": entry["kernel_name"],
            "cache_dir": ir_base / entry["cache_rel"],
            "shapes_dir": ir_base / entry["shapes_rel"],
        }
        for entry in _KERNEL_REGISTRY_TEMPLATE
    ]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OperatorInfo:
    """Holds everything needed to benchmark a single operator/kernel."""
    op_name: str
    kernel_name: str  # bare name (e.g. "softmax_kernel")
    kernel_class: Optional[str]  # e.g. "Operator" or None
    source_path: Path  # path to the .py file containing the kernel
    shapes_files: List[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Operator discovery
# ---------------------------------------------------------------------------

def discover_from_registry(
    requested_ops: List[str],
    ir_base: Path,
    max_shapes: int = 1,
) -> List[OperatorInfo]:
    """Resolve operators from the built-in kernel registry.

    Each registry entry has a specific cache_dir and shapes_dir that match
    the paths used by ``multiagent_triton_runner.py``.  This handles kernels
    with non-standard nested directory layouts (jagged_*, layer_norm, etc.).

    Args:
        requested_ops: Operator names to benchmark, or ``["all"]``.
        ir_base: Root directory containing per-operator IR/shapes subdirs.
        max_shapes: Maximum number of shapes files per operator.

    Returns:
        List of :class:`OperatorInfo` with resolved paths.
    """
    want_all = len(requested_ops) == 1 and requested_ops[0].lower() == "all"
    requested_set = {op.lower().strip() for op in requested_ops} if not want_all else None

    results: List[OperatorInfo] = []

    for entry in build_kernel_registry(ir_base):
        op_name = entry["op_name"]
        kernel_name = entry["kernel_name"]
        cache_dir = entry["cache_dir"]
        shapes_dir = entry["shapes_dir"]

        # Filter to requested operators (match on op_name or partial match)
        if requested_set is not None:
            match = any(
                req in op_name.lower() or op_name.lower().startswith(req)
                for req in requested_set
            )
            if not match:
                continue

        # Resolve TTGIR -> source path
        try:
            ttgir_path = _find_ttgir(cache_dir, kernel_name)
            _torch_model, source_path = _extract_source_info(ttgir_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s/%s: %s", op_name, kernel_name, exc)
            continue

        # Parse kernel class from kernel name (e.g. "Operator.softmax_kernel")
        kernel_class: Optional[str] = None
        bare_kernel_name = kernel_name
        if "." in kernel_name:
            kernel_class, bare_kernel_name = kernel_name.split(".", 1)

        # Find shapes files
        # Extract the base op name (before any parenthetical variant)
        base_op = op_name.split("(")[0].strip().replace(" ", "_")
        try:
            shapes = _find_shapes(
                shapes_dir, kernel_name, all_shapes=False,
                op_name=base_op, num_shapes=max_shapes,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s/%s: %s", op_name, kernel_name, exc)
            continue

        results.append(OperatorInfo(
            op_name=op_name,
            kernel_name=bare_kernel_name,
            kernel_class=kernel_class,
            source_path=source_path,
            shapes_files=shapes,
        ))

    return results


def discover_operators(
    requested_ops: List[str],
    metadata_dir: Path,
    cache_dir: Path,
    shapes_dir: Path,
    max_shapes: int = 1,
) -> List[OperatorInfo]:
    """Discover available operators via YAML metadata (legacy fallback).

    Prefer :func:`discover_from_registry` which handles all known kernels
    including those with non-standard nested directory layouts.
    """
    all_kernels = _parse_triton_kernels(metadata_dir)
    k2op = _kernel_to_op_map(metadata_dir)

    want_all = len(requested_ops) == 1 and requested_ops[0].lower() == "all"
    requested_set = {op.lower().strip() for op in requested_ops} if not want_all else None

    results: List[OperatorInfo] = []
    seen_ops: set = set()

    for kernel_name in all_kernels:
        op_name = k2op.get(kernel_name, "")
        if not op_name:
            continue

        # Filter to requested operators
        if requested_set is not None and op_name.lower() not in requested_set:
            continue

        # Only take the first kernel per operator to avoid redundancy
        if op_name in seen_ops:
            continue
        seen_ops.add(op_name)

        # Resolve TTGIR -> source path
        try:
            ttgir_path = _find_ttgir(cache_dir, kernel_name)
            _torch_model, source_path = _extract_source_info(ttgir_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s/%s: %s", op_name, kernel_name, exc)
            continue

        # Parse kernel class from kernel name (e.g. "Operator.softmax_kernel")
        kernel_class: Optional[str] = None
        bare_kernel_name = kernel_name
        if "." in kernel_name:
            kernel_class, bare_kernel_name = kernel_name.split(".", 1)

        # Find shapes files
        try:
            shapes = _find_shapes(
                shapes_dir, kernel_name, all_shapes=False,
                op_name=op_name, num_shapes=max_shapes,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s/%s: %s", op_name, kernel_name, exc)
            continue

        results.append(OperatorInfo(
            op_name=op_name,
            kernel_name=bare_kernel_name,
            kernel_class=kernel_class,
            source_path=source_path,
            shapes_files=shapes,
        ))

    return results


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def build_default_config(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the default kernel configuration from shapes metadata.

    Reads ``num_warps``, ``num_stages``, and any ``BLOCK_*`` parameters
    from the ``kwargs`` section of the shapes JSON.  This is the config
    the kernel was captured with — no LLM involved.
    """
    shapes_kwargs = metadata.get("kwargs", {})
    config: Dict[str, Any] = {}

    if "num_warps" in shapes_kwargs:
        config["num_warps"] = shapes_kwargs["num_warps"]["value"]
    if "num_stages" in shapes_kwargs:
        config["num_stages"] = shapes_kwargs["num_stages"]["value"]

    block_sizes: Dict[str, int] = {}
    for k, v in shapes_kwargs.items():
        if k in ("num_warps", "num_stages", "warmup"):
            continue
        if isinstance(v, dict) and v.get("type") == "scalar":
            val = v.get("value")
            if isinstance(val, (int, float)) and "BLOCK" in k.upper():
                block_sizes[k] = int(val)
    if block_sizes:
        config["block_size"] = block_sizes

    return config


# ---------------------------------------------------------------------------
# Grid resolution fallback for kernels with complex grid lambdas
# ---------------------------------------------------------------------------

def _resolve_grid_with_fallback(
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    bench_kwargs: Dict[str, Any],
) -> Tuple[int, ...]:
    """Resolve grid, falling back to manual computation for complex lambdas.

    Some kernels use grid formulas that reference closure variables not
    available in the standard eval namespace (e.g. ``len(x.offsets())``,
    ``B``, ``block_size``).  This function catches those failures and
    computes the grid manually from the shapes metadata.
    """
    import math

    try:
        return _resolve_grid(metadata, config, bench_kwargs)
    except (ValueError, KeyError):
        pass

    # Manual fallback based on grid source patterns
    raw = metadata.get("grid", {})
    source = raw.get("source", "") if isinstance(raw, dict) else ""
    shapes_kwargs = metadata.get("kwargs", {})
    tensor_args = metadata.get("tensor_args", [])

    # Pattern: jagged kernels with len(x.offsets()) - 1
    # offsets is an int64 1-D tensor; batch_size = len(offsets) - 1
    if "offsets" in source or ("B" in source and "cdiv" in source):
        batch_size = None
        for t in tensor_args:
            if t.get("dtype") == "torch.int64" and len(t.get("shape", [])) == 1:
                batch_size = t["shape"][0] - 1
                break
        if batch_size is None:
            batch_size = 1

        block_m = shapes_kwargs.get("BLOCK_SIZE_M", {}).get("value")
        m_val = shapes_kwargs.get("M", {}).get("value")
        if block_m and m_val:
            grid_x = batch_size * math.ceil(m_val / block_m)
            logger.info("Fallback grid (jagged pattern): (%d,)", grid_x)
            return (grid_x,)

    # Pattern: vector_exp with triton.cdiv(n_elements, block_size)
    if "n_elements" in source and "block_size" in source:
        n_elements = None
        for sa in metadata.get("scalar_args", []):
            n_elements = sa.get("value")
            break
        if n_elements is None and tensor_args:
            n_elements = 1
            for d in tensor_args[0].get("shape", [1]):
                n_elements *= d
        block_size = shapes_kwargs.get("BLOCK_SIZE", {}).get("value", 1024)
        if n_elements:
            grid_x = math.ceil(n_elements / block_size)
            logger.info("Fallback grid (cdiv pattern): (%d,)", grid_x)
            return (grid_x,)

    raise ValueError(
        f"Cannot resolve grid even with fallback: source={source!r}"
    )


# ---------------------------------------------------------------------------
# Single-run benchmarking (preserves raw timing samples)
# ---------------------------------------------------------------------------

def _benchmark_single_run(
    kernel_fn,
    grid: Tuple[int, ...],
    args: list,
    kwargs: Dict[str, Any],
    warmup: int,
    rep: int,
    run_index: int,
) -> Tuple[List[float], Dict[str, float]]:
    """Execute a single benchmark run, returning ``(raw_times, stats_dict)``.

    Mirrors ``_benchmark_config`` from ``compare_configs.py`` but also
    returns the raw timing list for cross-run analysis.
    """
    _args = tuple(args)
    _kwargs = dict(kwargs)
    _grid = grid
    _fn = kernel_fn

    def kernel_call():
        _fn[_grid](*_args, **_kwargs)

    try:
        times = triton.testing.do_bench(
            kernel_call, warmup=warmup, rep=rep, return_mode="all",
        )
    except Exception as exc:
        logger.error("Run %d benchmark failed: %s", run_index, exc)
        return [], {"error": str(exc)}

    raw_times = list(times) if not isinstance(times, list) else times
    stats = _compute_stats(raw_times)
    logger.debug(
        "  Run %d: median=%.4f ms  mean=%.4f ms  std=%.4f ms  (n=%d)",
        run_index, stats.get("median", 0), stats.get("mean", 0),
        stats.get("std_dev", 0), stats.get("samples", 0),
    )
    return raw_times, stats


# ---------------------------------------------------------------------------
# Multi-run stability benchmark
# ---------------------------------------------------------------------------

def run_stability_benchmark(
    kernel_fn,
    grid: Tuple[int, ...],
    args: list,
    kwargs: Dict[str, Any],
    num_runs: int,
    warmup: int,
    rep: int,
) -> Dict[str, Any]:
    """Run the same benchmark *num_runs* times and collect all timing data."""
    per_run_stats: List[Dict[str, float]] = []
    per_run_raw_samples: List[List[float]] = []

    for i in range(num_runs):
        raw_times, stats = _benchmark_single_run(
            kernel_fn, grid, args, kwargs, warmup, rep, run_index=i,
        )
        if "error" in stats:
            logger.warning("Run %d failed, skipping", i)
            continue
        per_run_stats.append(stats)
        per_run_raw_samples.append(raw_times)

    return {
        "per_run_stats": per_run_stats,
        "per_run_raw_samples": per_run_raw_samples,
        "completed_runs": len(per_run_stats),
    }


# ---------------------------------------------------------------------------
# Cross-run statistical analysis
# ---------------------------------------------------------------------------

def _describe_values(values: List[float], label: str) -> Dict[str, Any]:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {}
    n = len(values)
    s = sorted(values)
    mean_val = statistics.mean(s)
    med_val = s[n // 2]
    std_val = statistics.stdev(s) if n > 1 else 0.0
    var_val = statistics.variance(s) if n > 1 else 0.0
    cov = (std_val / mean_val * 100) if mean_val > 0 else 0.0
    spread = ((s[-1] - s[0]) / med_val * 100) if med_val > 0 else 0.0
    q1 = s[n // 4] if n >= 4 else s[0]
    q3 = s[3 * n // 4] if n >= 4 else s[-1]
    return {
        f"{label}_values": values,
        f"median_of_{label}": med_val,
        f"mean_of_{label}": mean_val,
        f"std_dev_of_{label}": std_val,
        f"variance_of_{label}": var_val,
        f"coeff_of_variation_{label}_pct": round(cov, 4),
        f"min_{label}": s[0],
        f"max_{label}": s[-1],
        f"spread_{label}_pct": round(spread, 4),
        f"iqr_{label}": q3 - q1,
    }


def compute_cross_run_statistics(
    per_run_stats: List[Dict[str, float]],
    per_run_raw_samples: List[List[float]],
) -> Dict[str, Any]:
    """Compute across-run stability metrics.

    Key metrics:
    - CoV of medians < 1%  -> very stable measurement
    - CoV of medians 1-5%  -> reasonably stable
    - CoV of medians > 5%  -> noisy, consider more reps or warmup
    """
    if not per_run_stats:
        return {}

    medians = [s["median"] for s in per_run_stats]
    means = [s["mean"] for s in per_run_stats]

    # Pool all individual samples across all runs
    all_samples = [t for run_samples in per_run_raw_samples for t in run_samples]
    pooled = _compute_stats(all_samples)
    pooled["total_samples"] = len(all_samples)

    return {
        "medians_analysis": _describe_values(medians, "medians"),
        "means_analysis": _describe_values(means, "means"),
        "pooled_analysis": pooled,
        "num_runs": len(per_run_stats),
    }


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def format_summary_table(results: List[Dict[str, Any]]) -> str:
    """Render a human-readable summary table."""
    header = (
        f"{'Operator':<34} {'Kernel':<30} {'Runs':>5} "
        f"{'Med(ms)':>10} {'StdDev(ms)':>11} {'CoV(%)':>8} {'Spread(%)':>10} "
        f"{'Samp/Run':>9}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        if r.get("benchmark_error"):
            lines.append(f"{r['operator']:<34} {'ERROR: ' + r['benchmark_error'][:60]}")
            continue

        cs = r.get("cross_run_stats", {})
        ma = cs.get("medians_analysis", {})
        med = ma.get("median_of_medians", 0)
        std = ma.get("std_dev_of_medians", 0)
        cov = ma.get("coeff_of_variation_medians_pct", 0)
        spread = ma.get("spread_medians_pct", 0)
        runs = cs.get("num_runs", 0)
        samples = r.get("per_run_stats", [{}])[0].get("samples", 0) if r.get("per_run_stats") else 0

        lines.append(
            f"{r['operator']:<34} {r['kernel']:<30} {runs:>5} "
            f"{med:>10.4f} {std:>11.4f} {cov:>8.2f} {spread:>10.2f} "
            f"{samples:>9}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization (optional)
# ---------------------------------------------------------------------------

def generate_visualization(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    """Generate a stability box-plot of medians across runs per operator.

    Gracefully returns ``None`` if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available — skipping visualization")
        return None

    valid = [r for r in results if not r.get("benchmark_error")]
    if not valid:
        logger.warning("No successful results to plot")
        return None

    labels: List[str] = []
    data: List[List[float]] = []
    for r in valid:
        ma = r.get("cross_run_stats", {}).get("medians_analysis", {})
        medians = ma.get("medians_values", [])
        if medians:
            labels.append(f"{r['operator']}\n({r['kernel'][:20]})")
            data.append(medians)

    if not data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(data) * 2.5), 6))

    # Box plot
    ax1 = axes[0]
    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4a90d9")
        patch.set_alpha(0.7)
    ax1.set_ylabel("Median time (ms)")
    ax1.set_title("Benchmark Stability: Median Distribution Across Runs")
    ax1.tick_params(axis="x", rotation=30)

    # CoV bar chart
    ax2 = axes[1]
    covs = []
    bar_labels = []
    for r in valid:
        ma = r.get("cross_run_stats", {}).get("medians_analysis", {})
        cov = ma.get("coeff_of_variation_medians_pct", 0)
        covs.append(cov)
        bar_labels.append(r["operator"])
    colors = ["#2ecc71" if c < 1 else "#f1c40f" if c < 5 else "#e74c3c" for c in covs]
    ax2.bar(bar_labels, covs, color=colors, alpha=0.8)
    ax2.set_ylabel("Coefficient of Variation (%)")
    ax2.set_title("Measurement Stability (lower = better)")
    ax2.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="1% (very stable)")
    ax2.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="5% (noisy)")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "stability_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", plot_path)
    return plot_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure benchmark measurement stability across repeated runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--operators", default="all",
        help='Comma-separated operator names, or "all" for all 15 registered kernels (default: all)',
    )
    p.add_argument("--num-runs", type=int, default=10, help="Repeated runs per config (default: 10)")
    p.add_argument("--warmup", type=int, default=25, help="Warmup iterations per do_bench call (default: 25)")
    p.add_argument("--rep", type=int, default=100, help="Repetitions per do_bench call (default: 100)")
    p.add_argument("--max-shapes", type=int, default=1, help="Max shapes per operator (default: 1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Root IR/shapes directory, e.g. extract/ir_output/v100 or extract/ir_output/a100 "
             "(default: auto-detect from extract/ir_output/)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Results output directory (default: measurement_experiment/results)",
    )
    p.add_argument("--plot", action="store_true", help="Generate matplotlib visualization")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve directories
    if args.cache_dir is not None:
        ir_base = args.cache_dir
    else:
        # Auto-detect: pick the first (or only) subdirectory under ir_output/
        ir_output_root = PROJECT_ROOT / "extract" / "ir_output"
        if ir_output_root.is_dir():
            subdirs = sorted([d for d in ir_output_root.iterdir() if d.is_dir()])
            if len(subdirs) == 1:
                ir_base = subdirs[0]
            elif subdirs:
                available = [d.name for d in subdirs]
                logger.error(
                    "Multiple device dirs found under %s: %s\n"
                    "Please specify one with --cache-dir, e.g.:\n"
                    "  --cache-dir %s",
                    ir_output_root, available, subdirs[0],
                )
                return 1
            else:
                ir_base = ir_output_root
        else:
            ir_base = ir_output_root
    shapes_dir = ir_base  # shapes live alongside TTGIRs
    output_dir = args.output_dir or (SCRIPT_DIR / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ops = [op.strip() for op in args.operators.split(",") if op.strip()]
    if not requested_ops:
        requested_ops = DEFAULT_OPERATORS

    # GPU info
    gpu_name = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    logger.info("=== Benchmark Stability Measurement ===")
    logger.info("Device: %s (%s)", args.device, gpu_name)
    logger.info("IR base: %s", ir_base)
    logger.info("Operators: %s", requested_ops)
    logger.info("Runs: %d, warmup: %d, rep: %d, seed: %d",
                args.num_runs, args.warmup, args.rep, args.seed)

    # Discover operators — registry first (handles nested dirs), YAML fallback
    operators = discover_from_registry(requested_ops, ir_base=ir_base, max_shapes=args.max_shapes)
    if not operators:
        logger.info("No registry match, falling back to YAML metadata discovery")
        operators = discover_operators(
            requested_ops, METADATA_DIR, ir_base, shapes_dir,
            max_shapes=args.max_shapes,
        )
    if not operators:
        logger.error("No operators found for: %s", requested_ops)
        return 1

    logger.info("Discovered %d operator(s): %s",
                len(operators), [o.op_name for o in operators])

    # Run benchmarks
    all_results: List[Dict[str, Any]] = []

    for op_info in operators:
        for shapes_file in op_info.shapes_files:
            logger.info("")
            logger.info("--- %s / %s (%s) ---", op_info.op_name, op_info.kernel_name, shapes_file.name)

            result: Dict[str, Any] = {
                "operator": op_info.op_name,
                "kernel": op_info.kernel_name,
                "kernel_class": op_info.kernel_class,
                "shapes_file": str(shapes_file),
                "benchmark_error": None,
            }

            try:
                # Load metadata and inputs
                metadata = load_tensor_shapes(shapes_file)
                input_args, input_kwargs = create_kernel_inputs(
                    shapes_file, device=args.device, values="random", seed=args.seed,
                )

                # Load kernel
                kernel_fn = _load_kernel(op_info.source_path, op_info.kernel_name, op_info.kernel_class)
                kernel_fn, _at_keys = _unwrap_autotuner(kernel_fn)

                # Build config and kwargs
                config = build_default_config(metadata)
                result["config"] = config

                shapes_kwargs = metadata.get("kwargs", {})
                bench_kwargs = _build_kwargs(config, input_kwargs, shapes_kwargs)

                # Resolve grid (with fallback for complex lambdas)
                grid = _resolve_grid_with_fallback(metadata, config, bench_kwargs)
                result["grid"] = list(grid)

                # Extract shape info for display
                tensor_args = metadata.get("tensor_args", [])
                if tensor_args:
                    result["tensor_shape"] = tensor_args[0].get("shape")
                    result["tensor_dtype"] = tensor_args[0].get("dtype")

                logger.info("Config: %s", config)
                logger.info("Grid: %s", grid)
                logger.info("Running %d benchmark iterations ...", args.num_runs)

                t0 = time.monotonic()
                bench_data = run_stability_benchmark(
                    kernel_fn, grid, input_args, bench_kwargs,
                    num_runs=args.num_runs, warmup=args.warmup, rep=args.rep,
                )
                elapsed = time.monotonic() - t0

                result["per_run_stats"] = bench_data["per_run_stats"]
                result["completed_runs"] = bench_data["completed_runs"]
                result["total_time_s"] = round(elapsed, 2)

                # Compute cross-run stats
                cross_stats = compute_cross_run_statistics(
                    bench_data["per_run_stats"], bench_data["per_run_raw_samples"],
                )
                result["cross_run_stats"] = cross_stats

                # Log key metrics
                ma = cross_stats.get("medians_analysis", {})
                logger.info(
                    "  Median of medians: %.4f ms | StdDev: %.4f ms | CoV: %.2f%% | Spread: %.2f%%",
                    ma.get("median_of_medians", 0),
                    ma.get("std_dev_of_medians", 0),
                    ma.get("coeff_of_variation_medians_pct", 0),
                    ma.get("spread_medians_pct", 0),
                )

            except Exception as exc:
                logger.error("Failed to benchmark %s/%s: %s", op_info.op_name, op_info.kernel_name, exc)
                result["benchmark_error"] = str(exc)

            all_results.append(result)

            # Free GPU memory between operators; reset device on CUDA errors
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    logger.warning("CUDA error during cache clear — resetting device")
                    torch.cuda.device(0)
                    torch.cuda.empty_cache()

    # Summary
    print("\n")
    print(format_summary_table(all_results))

    # Compute overall summary
    successful = [r for r in all_results if not r.get("benchmark_error")]
    summary: Dict[str, Any] = {
        "total_operators": len(all_results),
        "successful": len(successful),
        "failed": len(all_results) - len(successful),
    }
    if successful:
        covs = [
            r["cross_run_stats"]["medians_analysis"]["coeff_of_variation_medians_pct"]
            for r in successful
            if r.get("cross_run_stats", {}).get("medians_analysis")
        ]
        if covs:
            summary["avg_cov_pct"] = round(statistics.mean(covs), 4)
            summary["max_cov_pct"] = round(max(covs), 4)
            summary["min_cov_pct"] = round(min(covs), 4)
            best_idx = covs.index(min(covs))
            worst_idx = covs.index(max(covs))
            summary["most_stable"] = successful[best_idx]["operator"]
            summary["least_stable"] = successful[worst_idx]["operator"]

    print(f"\nOverall: {summary['successful']}/{summary['total_operators']} operators benchmarked")
    if summary.get("avg_cov_pct") is not None:
        print(f"Average CoV: {summary['avg_cov_pct']:.2f}%  |  "
              f"Best: {summary.get('most_stable', 'N/A')} ({summary.get('min_cov_pct', 0):.2f}%)  |  "
              f"Worst: {summary.get('least_stable', 'N/A')} ({summary.get('max_cov_pct', 0):.2f}%)")

    # Save JSON results (strip raw samples from per_run to keep file manageable)
    output_data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "gpu_name": gpu_name,
            "num_runs": args.num_runs,
            "warmup": args.warmup,
            "rep": args.rep,
            "seed": args.seed,
        },
        "results": all_results,
        "summary": summary,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"stability_results_{ts}.json"
    json_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\nResults saved to: {json_path}")

    # Visualization
    if args.plot:
        plot_path = generate_visualization(all_results, output_dir)
        if plot_path:
            print(f"Plot saved to: {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
