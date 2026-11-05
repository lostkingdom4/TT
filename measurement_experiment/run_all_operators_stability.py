#!/usr/bin/env python3
"""Run stability benchmarks on ALL discovered operators using the batch runner's
discovery mechanism.

Combines the batch_runner's automatic kernel discovery (scans IR output dirs for
all operators/kernels with shapes files and TTGIRs) with the stability benchmark
infrastructure from measure_benchmark_stability.py.

For each discovered kernel:
  1. Discovers shapes files and TTGIR via batch_runner.discover_kernels()
  2. Resolves source path via _extract_source_info()
  3. Loads kernel, builds config/kwargs/grid from shapes metadata
  4. Runs N benchmark iterations using triton.testing.do_bench (warmup + rep)
  5. Computes per-run and cross-run stability statistics
  6. Reports latency (median, mean, std_dev, CoV) for every operator

Usage:
    # All operators, 5 runs each (quick)
    python measurement_experiment/run_all_operators_stability.py --num-runs 5

    # All operators, 10 runs, 25 warmup, 100 rep (thorough)
    python measurement_experiment/run_all_operators_stability.py

    # Specific operators only
    python measurement_experiment/run_all_operators_stability.py --operators softmax,addmm

    # Specify GPU IR directory
    python measurement_experiment/run_all_operators_stability.py --gpu v100

    # Dry run: list discovered kernels without benchmarking
    python measurement_experiment/run_all_operators_stability.py --dry-run
"""

import argparse
import json
import logging
import multiprocessing
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import triton
import triton.testing

from batch_runner import classify_kernel_source, discover_kernels
from compare_configs import (
    _build_kwargs,
    _compute_stats,
    _load_kernel,
    _resolve_grid,
    _unwrap_autotuner,
)
from multiagent_triton_runner import _extract_source_info, _find_ttgir
from tensor_utils import create_kernel_inputs, load_tensor_shapes


# ---------------------------------------------------------------------------
# Operator-specific input fixups
# ---------------------------------------------------------------------------

def fixup_inputs(operator: str, kernel_name: str, args: list,
                 metadata: dict, device: str = "cuda") -> list:
    """Post-process generated inputs for kernels that need semantically valid data.

    TritonBench operators like jagged_*, embedding, and mamba2 require inputs
    with specific value constraints that generic random generation violates.
    """
    tensor_args = metadata.get("tensor_args", [])

    # --- Jagged operators: offsets tensor must be sorted cumulative int64 ---
    if operator.startswith("jagged"):
        for i, tinfo in enumerate(tensor_args):
            if (tinfo["dtype"] == "torch.int64"
                    and len(tinfo["shape"]) == 1
                    and i < len(args)):
                # This is the offsets tensor: generate valid cumulative offsets
                n = tinfo["shape"][0]  # number of offsets (batch_size + 1)
                # Find the values tensor (largest float tensor, typically arg 0)
                total_elems = None
                for j, t2 in enumerate(tensor_args):
                    if t2["dtype"] != "torch.int64" and len(t2["shape"]) >= 1:
                        total_elems = t2["shape"][0]
                        break
                if total_elems and n > 1:
                    # Generate n-1 random lengths that sum to total_elems
                    batch = n - 1
                    avg_len = total_elems // batch
                    import random
                    lengths = [max(1, int(random.gauss(avg_len, avg_len // 3)))
                               for _ in range(batch)]
                    # Scale to match total_elems
                    scale = total_elems / sum(lengths)
                    lengths = [int(l * scale) for l in lengths]
                    lengths[-1] = total_elems - sum(lengths[:-1])
                    offsets = [0]
                    for l in lengths:
                        offsets.append(offsets[-1] + max(0, l))
                    args[i] = torch.tensor(offsets, dtype=torch.int64, device=device)

    # --- Cross-entropy / NLL loss: label tensors must be valid class indices ---
    elif operator in ("cross_entropy", "fused_linear_cross_entropy"):
        for i, tinfo in enumerate(tensor_args):
            if tinfo["dtype"] == "torch.int64" and i < len(args):
                # Labels: must be in [0, num_classes) or -100 (ignore_index).
                # Infer num_classes from the largest float tensor's last dim.
                num_classes = 1000  # safe default
                for t2 in tensor_args:
                    if t2["dtype"] != "torch.int64" and len(t2["shape"]) >= 2:
                        num_classes = t2["shape"][-1]
                        break
                shape = tuple(tinfo["shape"])
                args[i] = torch.randint(0, num_classes, shape,
                                        dtype=torch.int64, device=device)

    # --- Embedding: index tensor must be in [0, vocab_size) ---
    elif operator == "embedding":
        # Find the embedding weight tensor (2D float tensor) to get vocab_size
        vocab_size = None
        for tinfo in tensor_args:
            if (len(tinfo["shape"]) == 2
                    and tinfo["dtype"] != "torch.int64"):
                vocab_size = tinfo["shape"][0]
                break
        if vocab_size:
            for i, tinfo in enumerate(tensor_args):
                if tinfo["dtype"] == "torch.int64" and i < len(args):
                    shape = tuple(tinfo["shape"])
                    args[i] = torch.randint(0, vocab_size, shape,
                                            dtype=torch.int64, device=device)

    # --- mamba2: ensure no NaN/Inf and proper value ranges ---
    elif operator.startswith("mamba2"):
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                # Use small random values to avoid numerical issues
                args[i] = torch.rand_like(arg) * 0.1

    return args


# ---------------------------------------------------------------------------
# Shapes-file / kernel numel matching for inductor kernels
# ---------------------------------------------------------------------------

def _extract_hardcoded_numels(kernel_py: Path) -> Dict[str, int]:
    """Parse hardcoded size constants from an inductor kernel .py.

    Captures:
    - ``xnumel = N``, ``rnumel = N``, ``ynumel = N`` (reduction/pointwise)
    - ``M = N``, ``N = N``, ``K = N`` (template matmul kernels like triton_mm)
    """
    import re
    numels: Dict[str, int] = {}
    try:
        src = kernel_py.read_text()
        # xnumel/rnumel/ynumel — standard inductor pattern
        for m in re.finditer(r'\b(\w*numel)\s*=\s*(\d+)', src):
            numels[m.group(1)] = int(m.group(2))
        # M/N/K — template matmul pattern (only inside function body)
        if not numels:
            for m in re.finditer(r'^\s+([MNK])\s*=\s*(\d+)', src, re.MULTILINE):
                numels[m.group(1)] = int(m.group(2))
    except OSError:
        pass
    return numels


def _extract_kernel_strides(kernel_py: Path) -> List[int]:
    """Extract large integer constants used as strides/sizes in kernel loads/stores.

    Inductor kernels embed strides like ``tl.load(ptr + (idx + 65536*x0))``
    where 65536 corresponds to a tensor dimension.  These must match the
    shapes file's tensor dimensions.
    """
    import re
    strides: set = set()
    try:
        src = kernel_py.read_text()
        # Match patterns like: 65536*x0, 536870912*x0, + 4096*r1
        for m in re.finditer(r'\b(\d{4,})\s*\*\s*[xry]\d', src):
            strides.add(int(m.group(1)))
    except OSError:
        pass
    return sorted(strides)


def _shapes_file_matches_kernel(shapes_file: Path, numels: Dict[str, int],
                                kernel_strides: Optional[List[int]] = None) -> bool:
    """Check if a shapes file's dimensions match the kernel's hardcoded constants.

    Validates both:
    1. Scalar args match xnumel/rnumel (or tensor dims match M/N/K)
    2. Tensor dimensions can accommodate the kernel's hardcoded strides
    """
    if not numels:
        return True

    import json
    try:
        data = json.load(open(shapes_file))
    except (json.JSONDecodeError, OSError):
        return False

    scalar_vals = [s["value"] for s in data.get("scalar_args", [])]
    tensor_dims: List[int] = []
    for t in data.get("tensor_args", []):
        tensor_dims.extend(t.get("shape", []))
    # Also include products of dimensions (total element counts)
    tensor_elems: List[int] = []
    for t in data.get("tensor_args", []):
        shape = t.get("shape", [])
        if shape:
            prod = 1
            for d in shape:
                prod *= d
            tensor_elems.append(prod)

    # Also include partial products of adjacent dimensions (strides)
    tensor_strides: List[int] = []
    for t in data.get("tensor_args", []):
        shape = t.get("shape", [])
        # Compute row-major strides: product of trailing dims
        for start in range(len(shape)):
            prod = 1
            for d in shape[start:]:
                prod *= d
                tensor_strides.append(prod)

    all_known = set(scalar_vals) | set(tensor_dims) | set(tensor_elems) | set(tensor_strides)

    # Check xnumel/rnumel/ynumel against scalar args
    numel_keys = {k for k in numels if "numel" in k}
    if numel_keys:
        for k in numel_keys:
            if numels[k] not in scalar_vals:
                return False

    # Check M/N/K against tensor dimensions
    mnk_keys = {k for k in numels if k in ("M", "N", "K")}
    if mnk_keys:
        for k in mnk_keys:
            if numels[k] not in tensor_dims:
                return False

    # Check kernel strides against tensor dimensions.
    # Strides can be tensor dims, element counts, or products of dims with
    # scalar args (e.g. 65536*8192).  We check if each stride is either
    # directly known or is a product of a known value and a numel.
    if kernel_strides:
        numel_vals = set(numels.values())
        for stride in kernel_strides:
            if stride in all_known:
                continue
            # Check if stride = known_val * numel_val
            found = False
            for nv in numel_vals:
                if nv > 0 and stride % nv == 0 and (stride // nv) in all_known:
                    found = True
                    break
            if not found:
                return False

    return True


def find_matching_shapes_file(
    shapes_dir: Path,
    kernel_name: str,
    kernel_source: Path,
    operator: str,
    max_shapes: int = 1,
) -> Optional[Path]:
    """Find a shapes file whose scalars and tensor dims match the kernel.

    For inductor kernels that bake xnumel/rnumel and strides into the source,
    we must use the shapes file that was captured for the *same* input size.
    """
    numels = _extract_hardcoded_numels(kernel_source)
    if not numels:
        return None  # Handwritten kernels — caller uses default discovery

    strides = _extract_kernel_strides(kernel_source)

    from multiagent_triton_runner import _find_shapes
    try:
        all_shapes = _find_shapes(
            shapes_dir, kernel_name, all_shapes=True, op_name=operator,
        )
    except FileNotFoundError:
        return None

    for sf in all_shapes:
        if _shapes_file_matches_kernel(sf, numels, strides):
            return sf

    return None


logger = logging.getLogger("all_operators_stability")


# ---------------------------------------------------------------------------
# Grid resolution with fallback (from measure_benchmark_stability.py)
# ---------------------------------------------------------------------------

def _resolve_grid_with_fallback(
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    bench_kwargs: Dict[str, Any],
) -> Tuple[int, ...]:
    """Resolve grid, falling back to manual computation for complex lambdas."""
    import math

    try:
        return _resolve_grid(metadata, config, bench_kwargs)
    except (ValueError, KeyError, NameError, TypeError):
        pass

    raw = metadata.get("grid", {})
    source = raw.get("source", "") if isinstance(raw, dict) else ""
    shapes_kwargs = metadata.get("kwargs", {})
    tensor_args = metadata.get("tensor_args", [])

    # Pattern: jagged kernels with len(x.offsets()) - 1
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
            return (grid_x,)

    # Pattern: triton.cdiv(n_elements, block_size)
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
            return (grid_x,)

    # Generic fallback: try to evaluate grid source with all available values
    # Build a namespace with all known scalar values
    ns: Dict[str, Any] = {}
    for k, v in shapes_kwargs.items():
        if isinstance(v, dict) and "value" in v:
            ns[k] = v["value"]
    ns.update(bench_kwargs)
    ns["triton"] = triton
    ns["math"] = __import__("math")

    if source:
        try:
            grid_val = eval(source, ns)
            if isinstance(grid_val, (list, tuple)):
                return tuple(int(x) for x in grid_val)
            return (int(grid_val),)
        except Exception:
            pass

    raise ValueError(f"Cannot resolve grid: source={source!r}")


# ---------------------------------------------------------------------------
# Config extraction (from measure_benchmark_stability.py)
# ---------------------------------------------------------------------------

def build_default_config(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the default kernel configuration from shapes metadata."""
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
# Single-run benchmarking
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
    """Execute a single benchmark run using triton.testing.do_bench."""
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
    """Run the same benchmark num_runs times and collect all timing data."""
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
# Cross-run statistics
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
    cov = (std_val / mean_val * 100) if mean_val > 0 else 0.0
    spread = ((s[-1] - s[0]) / med_val * 100) if med_val > 0 else 0.0
    return {
        f"{label}_values": values,
        f"median_of_{label}": med_val,
        f"mean_of_{label}": mean_val,
        f"std_dev_of_{label}": std_val,
        f"coeff_of_variation_{label}_pct": round(cov, 4),
        f"spread_{label}_pct": round(spread, 4),
    }


def compute_cross_run_statistics(
    per_run_stats: List[Dict[str, float]],
    per_run_raw_samples: List[List[float]],
) -> Dict[str, Any]:
    """Compute across-run stability metrics."""
    if not per_run_stats:
        return {}

    medians = [s["median"] for s in per_run_stats]
    means = [s["mean"] for s in per_run_stats]

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
# Summary table
# ---------------------------------------------------------------------------

def format_summary_table(results: List[Dict[str, Any]]) -> str:
    """Render a human-readable summary table."""
    header = (
        f"{'Operator':<22} {'Kernel':<35} {'Source':<6} {'Runs':>5} "
        f"{'Med(ms)':>10} {'Mean(ms)':>10} {'StdDev(ms)':>11} {'CoV(%)':>8} {'Spread(%)':>10} "
        f"{'Samp/Run':>9}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        src = r.get("kernel_source", "?")[:6]
        if r.get("benchmark_error"):
            err_msg = r["benchmark_error"][:50]
            lines.append(f"{r['operator']:<22} {r['kernel']:<35} {src:<6} ERROR: {err_msg}")
            continue

        cs = r.get("cross_run_stats", {})
        ma = cs.get("medians_analysis", {})
        med = ma.get("median_of_medians", 0)
        mean_val = ma.get("mean_of_medians", 0)
        std = ma.get("std_dev_of_medians", 0)
        cov = ma.get("coeff_of_variation_medians_pct", 0)
        spread = ma.get("spread_medians_pct", 0)
        runs = cs.get("num_runs", 0)
        samples = (
            r.get("per_run_stats", [{}])[0].get("samples", 0)
            if r.get("per_run_stats") else 0
        )

        lines.append(
            f"{r['operator']:<22} {r['kernel']:<35} {src:<6} {runs:>5} "
            f"{med:>10.4f} {mean_val:>10.4f} {std:>11.4f} {cov:>8.2f} {spread:>10.2f} "
            f"{samples:>9}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kernel resolution: batch_runner WorkItem -> benchmarkable kernel
# ---------------------------------------------------------------------------

def resolve_kernel_for_benchmark(
    operator: str,
    kernel_name: str,
    cache_dir: Path,
    shapes_dir: Path,
    device: str = "cuda",
    seed: int = 42,
    max_shapes: int = 1,
) -> Dict[str, Any]:
    """Resolve a discovered kernel into everything needed for benchmarking.

    Returns a dict with kernel_fn, grid, args, kwargs, config, shapes_file,
    or raises on failure.
    """
    # Find TTGIR and source path
    ttgir_path = _find_ttgir(cache_dir, kernel_name)
    _torch_model, source_path = _extract_source_info(ttgir_path)

    # Parse kernel class
    kernel_class: Optional[str] = None
    bare_kernel_name = kernel_name
    if "." in kernel_name:
        kernel_class, bare_kernel_name = kernel_name.split(".", 1)

    # Find shapes file(s) — use the first available
    from multiagent_triton_runner import _find_shapes
    shapes_files = _find_shapes(
        shapes_dir, kernel_name, all_shapes=False,
        op_name=operator, num_shapes=max_shapes,
    )
    shapes_file = shapes_files[0]

    # Load metadata and create inputs
    metadata = load_tensor_shapes(shapes_file)
    input_args, input_kwargs = create_kernel_inputs(
        shapes_file, device=device, values="random", seed=seed,
    )

    # Load and unwrap kernel
    kernel_fn = _load_kernel(source_path, bare_kernel_name, kernel_class)
    kernel_fn, _at_keys = _unwrap_autotuner(kernel_fn)

    # Build config and kwargs
    config = build_default_config(metadata)
    shapes_kwargs = metadata.get("kwargs", {})
    bench_kwargs = _build_kwargs(config, input_kwargs, shapes_kwargs)

    # Resolve grid
    grid = _resolve_grid_with_fallback(metadata, config, bench_kwargs)

    return {
        "kernel_fn": kernel_fn,
        "grid": grid,
        "args": input_args,
        "kwargs": bench_kwargs,
        "config": config,
        "shapes_file": str(shapes_file),
        "metadata": metadata,
        "source_path": str(source_path),
    }


# ---------------------------------------------------------------------------
# Subprocess worker: benchmark a single kernel in an isolated process
# ---------------------------------------------------------------------------

def _subprocess_benchmark_kernel(
    operator: str,
    kernel_name: str,
    cache_dir_str: str,
    shapes_dir_str: str,
    device: str,
    seed: int,
    max_shapes: int,
    num_runs: int,
    warmup: int,
    rep: int,
    result_file: str,
):
    """Run in a subprocess to isolate CUDA context. Writes result to a JSON file."""
    import json
    import logging
    import sys
    import time
    from pathlib import Path

    # Re-setup path and imports inside subprocess
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("subprocess_worker")

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
    from multiagent_triton_runner import _extract_source_info, _find_shapes, _find_ttgir
    from tensor_utils import create_kernel_inputs, load_tensor_shapes

    cache_dir = Path(cache_dir_str)
    shapes_dir = Path(shapes_dir_str)

    result: Dict[str, Any] = {
        "operator": operator,
        "kernel": kernel_name,
        "benchmark_error": None,
    }

    try:
        # Resolve kernel — try multiple TTGIR candidates to find correct source
        kernel_class = None
        bare_kernel_name = kernel_name
        if "." in kernel_name:
            kernel_class, bare_kernel_name = kernel_name.split(".", 1)

        def _try_load_kernel(src_path, kname, kclass):
            """Try multiple loading strategies for the kernel."""
            import importlib.util as ilu

            # Strategy 1: standard _load_kernel
            try:
                return _load_kernel(src_path, kname, kclass)
            except (AttributeError, ImportError):
                pass

            # Strategy 2: try with common class names (e.g. Operator.softmax_kernel)
            if not kclass:
                for cls_name in ["Operator"]:
                    try:
                        return _load_kernel(src_path, kname, cls_name)
                    except (AttributeError, ImportError):
                        pass

            # Strategy 3: direct file import (bypass package resolution)
            try:
                spec = ilu.spec_from_file_location("_direct_kernel", str(src_path))
                if spec and spec.loader:
                    mod = ilu.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if kclass:
                        cls = getattr(mod, kclass)
                        return getattr(cls, kname)
                    return getattr(mod, kname)
            except (AttributeError, ImportError):
                pass

            raise AttributeError(f"Cannot find kernel {kname} in {src_path}")

        # Find TTGIR candidates — try multiple if first one resolves wrong source
        ttgir_candidates = sorted(cache_dir.rglob(f"*{kernel_name}*.ttgir"))
        source_path = None
        for ttgir_path in ttgir_candidates:
            try:
                _torch_model, candidate_src = _extract_source_info(ttgir_path)
                _try_load_kernel(candidate_src, bare_kernel_name, kernel_class)
                source_path = candidate_src
                break
            except (AttributeError, FileNotFoundError, ValueError):
                continue

        if source_path is None:
            raise FileNotFoundError(
                f"No valid TTGIR/source found for {kernel_name} in {cache_dir} "
                f"(tried {len(ttgir_candidates)} candidates)"
            )

        # Try to find a shapes file that matches the kernel's hardcoded numels.
        # Inductor kernels bake xnumel/rnumel into the .py source; using a
        # shapes file for a different input size causes OOB memory access.
        kernel_py_candidates = sorted(cache_dir.rglob(f"{kernel_name}.py"))
        kernel_py = kernel_py_candidates[0] if kernel_py_candidates else None

        matched_sf = None
        if kernel_py:
            matched_sf = find_matching_shapes_file(
                shapes_dir, kernel_name, kernel_py, operator,
            )

        if matched_sf:
            shapes_file = matched_sf
            log.info("  Matched shapes file to kernel numels: %s", shapes_file.name)
        else:
            shapes_files = _find_shapes(
                shapes_dir, kernel_name, all_shapes=False,
                op_name=operator, num_shapes=max_shapes,
            )
            shapes_file = shapes_files[0]

        metadata = load_tensor_shapes(shapes_file)
        input_args, input_kwargs = create_kernel_inputs(
            shapes_file, device=device, values="random", seed=seed,
        )

        # Fix up inputs for operators that need semantically valid data
        input_args = fixup_inputs(operator, kernel_name, input_args, metadata, device)

        kernel_fn = _try_load_kernel(source_path, bare_kernel_name, kernel_class)
        kernel_fn, _at_keys = _unwrap_autotuner(kernel_fn)

        config = build_default_config(metadata)
        shapes_kwargs = metadata.get("kwargs", {})
        bench_kwargs = _build_kwargs(config, input_kwargs, shapes_kwargs)
        grid = _resolve_grid_with_fallback(metadata, config, bench_kwargs)

        result["config"] = config
        result["grid"] = list(grid)
        result["shapes_file"] = str(shapes_file)
        result["source_path"] = str(source_path)

        tensor_args = metadata.get("tensor_args", [])
        if tensor_args:
            result["tensor_shape"] = tensor_args[0].get("shape")
            result["tensor_dtype"] = tensor_args[0].get("dtype")

        log.info("  Config: %s", config)
        log.info("  Grid: %s", grid)
        log.info("  Running %d benchmark runs ...", num_runs)

        # Benchmark
        _fn = kernel_fn
        _grid = grid
        _args = tuple(input_args)
        _kwargs = dict(bench_kwargs)

        def kernel_call():
            _fn[_grid](*_args, **_kwargs)

        per_run_stats = []
        per_run_raw_samples = []

        for i in range(num_runs):
            try:
                times = triton.testing.do_bench(
                    kernel_call, warmup=warmup, rep=rep, return_mode="all",
                )
                raw_times = list(times) if not isinstance(times, list) else times
                stats = _compute_stats(raw_times)
                per_run_stats.append(stats)
                per_run_raw_samples.append(raw_times)
                log.info(
                    "  Run %d: median=%.4f ms  mean=%.4f ms  (n=%d)",
                    i, stats.get("median", 0), stats.get("mean", 0),
                    stats.get("samples", 0),
                )
            except Exception as exc:
                log.error("  Run %d failed: %s", i, exc)
                # If CUDA error, break early — context is corrupted
                if "CUDA" in str(exc):
                    break

        result["per_run_stats"] = per_run_stats
        result["completed_runs"] = len(per_run_stats)

        if per_run_stats:
            # Cross-run statistics
            medians = [s["median"] for s in per_run_stats]
            means = [s["mean"] for s in per_run_stats]
            all_samples = [t for rs in per_run_raw_samples for t in rs]
            pooled = _compute_stats(all_samples)
            pooled["total_samples"] = len(all_samples)

            def describe(values, label):
                import statistics as st
                n = len(values)
                s = sorted(values)
                mean_v = st.mean(s)
                med_v = s[n // 2]
                std_v = st.stdev(s) if n > 1 else 0.0
                cov = (std_v / mean_v * 100) if mean_v > 0 else 0.0
                spread = ((s[-1] - s[0]) / med_v * 100) if med_v > 0 else 0.0
                return {
                    f"{label}_values": values,
                    f"median_of_{label}": med_v,
                    f"mean_of_{label}": mean_v,
                    f"std_dev_of_{label}": std_v,
                    f"coeff_of_variation_{label}_pct": round(cov, 4),
                    f"spread_{label}_pct": round(spread, 4),
                }

            result["cross_run_stats"] = {
                "medians_analysis": describe(medians, "medians"),
                "means_analysis": describe(means, "means"),
                "pooled_analysis": pooled,
                "num_runs": len(per_run_stats),
            }

            ma = result["cross_run_stats"]["medians_analysis"]
            log.info(
                "  Result: Median=%.4f ms | CoV=%.2f%% | Spread=%.2f%%",
                ma["median_of_medians"],
                ma["coeff_of_variation_medians_pct"],
                ma["spread_medians_pct"],
            )
        else:
            result["benchmark_error"] = "All runs failed"

    except Exception as exc:
        log.error("  FAILED: %s", exc)
        result["benchmark_error"] = str(exc)

    # Write result to file
    Path(result_file).write_text(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run stability benchmarks on ALL discovered operators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--gpu", default=None,
        help="GPU directory name under extract/ir_output/ (e.g. v100, a100). "
             "Auto-detected if only one exists.",
    )
    p.add_argument(
        "--operators", default=None,
        help="Comma-separated operator names to filter (default: all discovered)",
    )
    p.add_argument("--num-runs", type=int, default=10,
                    help="Repeated benchmark runs per kernel (default: 10)")
    p.add_argument("--warmup", type=int, default=25,
                    help="Warmup iterations per do_bench call (default: 25)")
    p.add_argument("--rep", type=int, default=100,
                    help="Repetitions per do_bench call (default: 100)")
    p.add_argument("--max-shapes", type=int, default=1,
                    help="Max shapes files per kernel (default: 1)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    p.add_argument("--device", default="cuda",
                    help="Device (default: cuda)")
    p.add_argument("--timeout", type=int, default=60,
                    help="Per-kernel timeout in seconds (default: 60)")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Results output directory (default: measurement_experiment/results)")
    p.add_argument("--max-kernels-per-op", type=int, default=None,
                    help="Max kernels per operator (default: all)")
    p.add_argument("--kernel-source", choices=["handwritten", "inductor", "all"],
                    default="all",
                    help="Filter by kernel source type (default: all)")
    p.add_argument("--dry-run", action="store_true",
                    help="List discovered kernels without benchmarking")
    p.add_argument("--verbose", action="store_true",
                    help="Verbose logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve IR directory
    ir_output_root = PROJECT_ROOT / "extract" / "ir_output"
    if args.gpu:
        ir_dir = ir_output_root / args.gpu
    else:
        subdirs = sorted([d for d in ir_output_root.iterdir() if d.is_dir()])
        if len(subdirs) == 1:
            ir_dir = subdirs[0]
        elif subdirs:
            available = [d.name for d in subdirs]
            logger.error(
                "Multiple device dirs found: %s. Use --gpu to specify.", available
            )
            return 1
        else:
            logger.error("No GPU directories found under %s", ir_output_root)
            return 1

    output_dir = args.output_dir or (SCRIPT_DIR / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU info
    gpu_name = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    logger.info("=" * 60)
    logger.info("ALL-OPERATOR STABILITY BENCHMARK")
    logger.info("=" * 60)
    logger.info("Device: %s (%s)", args.device, gpu_name)
    logger.info("IR directory: %s", ir_dir)
    logger.info("Runs: %d, warmup: %d, rep: %d, seed: %d",
                args.num_runs, args.warmup, args.rep, args.seed)

    # Discover ALL kernels using batch_runner's mechanism
    all_work_items = discover_kernels(ir_dir)

    # Filter out skipped items (no TTGIR)
    valid_items = [w for w in all_work_items if w.status != "skipped"]
    skipped_items = [w for w in all_work_items if w.status == "skipped"]

    # Filter by operator if requested
    if args.operators:
        requested = {op.strip().lower() for op in args.operators.split(",")}
        valid_items = [
            w for w in valid_items
            if w.operator.lower() in requested
        ]

    # Filter by kernel source type
    if args.kernel_source != "all":
        valid_items = [
            w for w in valid_items
            if classify_kernel_source(w.kernel_name) == args.kernel_source
        ]

    # Limit kernels per operator
    if args.max_kernels_per_op:
        per_op_count: Dict[str, int] = {}
        filtered = []
        for w in valid_items:
            cnt = per_op_count.get(w.operator, 0)
            if cnt < args.max_kernels_per_op:
                filtered.append(w)
                per_op_count[w.operator] = cnt + 1
        valid_items = filtered

    # Special handling for jagged_mean: keep handwritten + 1 inductor variant
    # (jagged_mean has 1500+ inductor variants that are mostly redundant)
    jm_items = [w for w in valid_items if w.operator == "jagged_mean"]
    if len(jm_items) > 2:
        jm_hw = [w for w in jm_items if classify_kernel_source(w.kernel_name) == "handwritten"]
        jm_ind = [w for w in jm_items if classify_kernel_source(w.kernel_name) == "inductor"]
        jm_keep = jm_hw + jm_ind[:1]
        jm_keep_ids = {id(w) for w in jm_keep}
        valid_items = [w for w in valid_items if w.operator != "jagged_mean"] + jm_keep
        logger.info("jagged_mean: kept %d of %d kernels (all hw + 1 inductor)",
                     len(jm_keep), len(jm_items))

    logger.info("Discovered %d kernels (%d valid, %d skipped)",
                len(all_work_items), len(valid_items), len(skipped_items))

    if args.dry_run:
        print(f"\n{'Operator':<26} {'Kernel':<45} {'Source':<13} {'Shapes':>6} {'Status':<10}")
        print("-" * 108)
        for w in all_work_items:
            status = w.status if w.status == "skipped" else "ready"
            src = classify_kernel_source(w.kernel_name)
            print(f"{w.operator:<26} {w.kernel_name:<45} {src:<13} {w.num_shapes_available:>6} {status:<10}")
        hw = sum(1 for w in valid_items if classify_kernel_source(w.kernel_name) == "handwritten")
        ind = len(valid_items) - hw
        print(f"\nTotal: {len(all_work_items)} kernels "
              f"({len(valid_items)} ready: {hw} handwritten + {ind} inductor, "
              f"{len(skipped_items)} skipped)")
        return 0

    if not valid_items:
        logger.error("No valid kernels found to benchmark")
        return 1

    # Run benchmarks — each kernel in an isolated subprocess to prevent
    # CUDA context corruption from cascading across kernels
    all_results: List[Dict[str, Any]] = []
    total = len(valid_items)
    import tempfile

    for idx, work_item in enumerate(valid_items, 1):
        operator = work_item.operator
        kernel_name = work_item.kernel_name
        cache_dir = PROJECT_ROOT / work_item.cache_dir
        shapes_dir = PROJECT_ROOT / work_item.shapes_dir

        kernel_source = classify_kernel_source(kernel_name)
        logger.info("")
        logger.info("[%d/%d] %s / %s (%s)", idx, total, operator, kernel_name, kernel_source)

        # Use a temp file to pass results back from subprocess
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"bench_{operator}_",
            dir=str(output_dir), delete=False,
        ) as tmp:
            result_file = tmp.name

        t0 = time.monotonic()

        # Run benchmark in isolated subprocess (spawn for clean CUDA context)
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=_subprocess_benchmark_kernel,
            args=(
                operator, kernel_name,
                str(cache_dir), str(shapes_dir),
                args.device, args.seed, args.max_shapes,
                args.num_runs, args.warmup, args.rep,
                result_file,
            ),
        )
        p.start()
        p.join(timeout=args.timeout)

        elapsed = time.monotonic() - t0

        if p.is_alive():
            logger.warning("  Subprocess timed out after %ds — killing", args.timeout)
            p.kill()
            p.join(timeout=5)
            result = {
                "operator": operator,
                "kernel": kernel_name,
                "kernel_source": kernel_source,
                "num_shapes_available": work_item.num_shapes_available,
                "benchmark_error": f"Timeout after {args.timeout}s",
            }
        else:
            # Read result from file
            try:
                result = json.loads(Path(result_file).read_text())
                result["num_shapes_available"] = work_item.num_shapes_available
                result["kernel_source"] = kernel_source
                result["total_time_s"] = round(elapsed, 2)
            except Exception as exc:
                result = {
                    "operator": operator,
                    "kernel": kernel_name,
                    "kernel_source": kernel_source,
                    "num_shapes_available": work_item.num_shapes_available,
                    "benchmark_error": f"Failed to read subprocess result: {exc}",
                }

        # Clean up temp file
        try:
            Path(result_file).unlink()
        except OSError:
            pass

        # Log summary
        if result.get("benchmark_error"):
            logger.error("  FAILED: %s", result["benchmark_error"][:80])
        else:
            ma = result.get("cross_run_stats", {}).get("medians_analysis", {})
            logger.info(
                "  Median: %.4f ms | CoV: %.2f%% | Spread: %.2f%% [%.1fs]",
                ma.get("median_of_medians", 0),
                ma.get("coeff_of_variation_medians_pct", 0),
                ma.get("spread_medians_pct", 0),
                elapsed,
            )

        all_results.append(result)

    # Print summary table
    print("\n")
    print(format_summary_table(all_results))

    # Overall summary
    successful = [r for r in all_results if not r.get("benchmark_error")]
    failed = [r for r in all_results if r.get("benchmark_error")]

    summary: Dict[str, Any] = {
        "total_kernels_discovered": len(all_work_items),
        "total_benchmarked": len(all_results),
        "successful": len(successful),
        "failed": len(failed),
        "skipped_no_ttgir": len(skipped_items),
    }

    if successful:
        covs = [
            r["cross_run_stats"]["medians_analysis"]["coeff_of_variation_medians_pct"]
            for r in successful
            if r.get("cross_run_stats", {}).get("medians_analysis")
        ]
        medians = [
            r["cross_run_stats"]["medians_analysis"]["median_of_medians"]
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
        if medians:
            summary["avg_latency_ms"] = round(statistics.mean(medians), 4)
            summary["min_latency_ms"] = round(min(medians), 4)
            summary["max_latency_ms"] = round(max(medians), 4)

    # Per-source-type breakdown
    hw_success = [r for r in successful if r.get("kernel_source") == "handwritten"]
    ind_success = [r for r in successful if r.get("kernel_source") == "inductor"]
    hw_failed = [r for r in failed if r.get("kernel_source") == "handwritten"]
    ind_failed = [r for r in failed if r.get("kernel_source") == "inductor"]

    summary["handwritten_success"] = len(hw_success)
    summary["handwritten_failed"] = len(hw_failed)
    summary["inductor_success"] = len(ind_success)
    summary["inductor_failed"] = len(ind_failed)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {summary['successful']}/{summary['total_benchmarked']} kernels benchmarked successfully")
    print(f"  Handwritten: {len(hw_success)} OK, {len(hw_failed)} failed")
    print(f"  Inductor:    {len(ind_success)} OK, {len(ind_failed)} failed")
    if skipped_items:
        print(f"  Skipped (no TTGIR): {len(skipped_items)}")
    if failed:
        print(f"\nFailed kernels:")
        for r in failed:
            print(f"  [{r.get('kernel_source','?'):>12}] {r['operator']}/{r['kernel']}: {r['benchmark_error'][:60]}")

    if summary.get("avg_cov_pct") is not None:
        print(f"\nStability:")
        print(f"  Avg CoV: {summary['avg_cov_pct']:.2f}%")
        print(f"  Best:  {summary.get('most_stable', 'N/A')} ({summary.get('min_cov_pct', 0):.2f}%)")
        print(f"  Worst: {summary.get('least_stable', 'N/A')} ({summary.get('max_cov_pct', 0):.2f}%)")

    if summary.get("avg_latency_ms") is not None:
        print(f"\nLatency:")
        print(f"  Avg:  {summary['avg_latency_ms']:.4f} ms")
        print(f"  Min:  {summary['min_latency_ms']:.4f} ms")
        print(f"  Max:  {summary['max_latency_ms']:.4f} ms")

    print(f"{'=' * 60}")

    # Save JSON results
    output_data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "gpu_name": gpu_name,
            "ir_directory": str(ir_dir),
            "num_runs": args.num_runs,
            "warmup": args.warmup,
            "rep": args.rep,
            "seed": args.seed,
        },
        "results": all_results,
        "summary": summary,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"all_operators_stability_{ts}.json"
    json_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\nResults saved to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
