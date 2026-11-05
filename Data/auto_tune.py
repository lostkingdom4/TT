# Autotune Triton Kernels by Parsing Triton Source
#
# This script:
# - Parses each Triton file to detect grid dimensionality (1D/2D/3D)
# - Extracts configs from @triton.autotune if present
# - Otherwise synthesizes a parameter sweep from meta-parameter names in the source
# - Times each candidate config and records the best result
#
# Usage:
#   python autotune_triton.py --root /path/to/run_dir [--device cuda|xpu|cpu]
#
import os
import re
import sys
import ast
import json
import time
import types
import argparse
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import triton  # noqa: F401
except Exception as e:
    triton = None
    triton_import_error = e
else:
    triton_import_error = None

# Configure logging to write to both file and console
def setup_logging(log_file: str = "auto_tune.log"):
    """Set up logging to write to both file and console"""
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Write to file (overwrite)
            logging.StreamHandler(sys.stdout)         # Also write to console
        ]
    )
    return logging.getLogger(__name__)

# Global logger instance
logger = None

def log_info(msg: str):
    """Helper function to log info messages safely"""
    if logger:
        logger.info(msg)
    else:
        print(msg)

def log_warning(msg: str):
    """Helper function to log warning messages safely"""
    if logger:
        logger.warning(msg)
    else:
        print(f"WARNING: {msg}")

def log_error(msg: str):
    """Helper function to log error messages safely"""
    if logger:
        logger.error(msg)
    else:
        print(f"ERROR: {msg}")

def log_debug(msg: str):
    """Helper function to log debug messages safely"""
    if logger:
        logger.debug(msg)
    else:
        print(f"DEBUG: {msg}")

def _install_triton_grid_shim():
    """
    Some Inductor-generated Triton code does:
      from torch._inductor.runtime.triton_heuristics import grid
    On certain PyTorch versions, that symbol isn't exported.
    We inject a tiny shim so kernel[grid(...)](...) works.
    """
    mod_name = 'torch._inductor.runtime.triton_heuristics'
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        # Minimal grid: return a tuple of sizes, e.g. grid(256) -> (256,), grid(64,32)->(64,32)
        def grid(*sizes, **kwargs):
            # Accept grid(lambda META: ...) too; if a callable is passed, fall back to 1-D guess
            if sizes and callable(sizes[0]):
                # Best-effort: try (1,) as a harmless default; your entrypoint may override anyway
                return (1,)
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                return tuple(sizes[0])
            return tuple(sizes)
        m.grid = grid
        sys.modules[mod_name] = m

def dynamic_import_from_path(module_name: str, file_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def get_python_inputs(py_mod: types.ModuleType, device: torch.device) -> List[torch.Tensor]:
    if not hasattr(py_mod, "get_inputs"):
        raise AttributeError("Python module does not define get_inputs()")
    inputs = py_mod.get_inputs()
    return [obj.to(device) for obj in inputs if isinstance(obj, torch.Tensor)]


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def detect_grid_dim_from_source(src: str) -> List[Tuple[str, List[int]]]:
    """
    Infer grid dimensionality from Triton kernel launches.
    Returns:
    List of (kernel_name, [grid_dims])
    Example:
        'triton_poi_fused_add_0[1, 1, 1](...)' -> ("triton_poi_fused_add_0", [1, 1, 1])
        'foo[256](...)' -> ("foo", [256])
    """
    outputs = []

    # --- Case 1: kernel[dim1, dim2, dim3]( ... ) - the format in your example
    pattern_launch_new = re.compile(
        r'^\s*([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]\s*\([^)]*\)',
        re.MULTILINE
    )

    for kernel_name, grid_inside in pattern_launch_new.findall(src):
        # Clean up the grid dimensions - remove any extra whitespace
        grid_inside = grid_inside.strip()

        # Handle both single dimension and multiple dimensions
        if ',' in grid_inside:
            # Multiple dimensions like [1, 1, 1]
            parts = [p.strip() for p in grid_inside.split(',') if p.strip()]
        else:
            # Single dimension like [256]
            parts = [grid_inside]

        # Convert to integers, handling any potential expressions
        dims = []
        for part in parts:
            try:
                # Try to evaluate simple integer expressions
                if part.isdigit():
                    dims.append(int(part))
                else:
                    # For more complex expressions, try eval in safe context
                    # but for now, just try basic integer conversion
                    dims.append(int(part))
            except (ValueError, SyntaxError):
                log_warning(f"Could not parse grid dimension '{part}' for kernel {kernel_name}")
                # If we can't parse one dimension, skip this kernel
                dims = []
                break

        if dims:
            outputs.append((kernel_name, dims))
            log_info(f"Found kernel: {kernel_name} with grid dims: {dims}")

    # --- Case 2: Also look for the old grid(...) format for compatibility
    pattern_launch_old = re.compile(
        r'^\s*([A-Za-z_]\w*)\s*\[\s*grid\s*\((.*?)\)\s*\]\s*\([^)]*\)',
        re.MULTILINE
    )

    for kernel_name, grid_inside in pattern_launch_old.findall(src):
        parts = [p.strip() for p in grid_inside.split(',') if p.strip()]
        if parts:
            try:
                dims = [int(p) for p in parts]
                outputs.append((kernel_name, dims))
                log_info(f"Found kernel (old format): {kernel_name} with grid dims: {dims}")
            except ValueError:
                log_warning(f"Could not parse grid args in launch: {grid_inside}")

    return outputs


def extract_meta_param_names(src: str) -> List[str]:
    candidates = [
        "BLOCK_SIZE", "XBLOCK"
        "BLOCK_M", "BLOCK_N", "BLOCK_K",
        "GROUP_M",
        "num_warps", "num_stages",
    ]
    return [name for name in candidates if re.search(rf"\b{name}\b", src)]


def synthesize_grid_from_names(names: List[str]) -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []
    if "BLOCK_M" in names or "BLOCK_N" in names or "BLOCK_K" in names:
        BM = [16, 32, 64, 128] if "BLOCK_M" in names else [None]
        BN = [16, 32, 64, 128] if "BLOCK_N" in names else [None]
        BK = [16, 32, 64] if "BLOCK_K" in names else [None]
        WARPS = [1, 2, 4, 8] if "num_warps" in names else [None]
        STAGES = [1, 2, 3, 4] if "num_stages" in names else [None]
        GM = [1, 2, 4, 8] if "GROUP_M" in names else [None]
        for bm in (BM if BM != [None] else [None]):
            for bn in (BN if BN != [None] else [None]):
                for bk in (BK if BK != [None] else [None]):
                    for w in (WARPS if WARPS != [None] else [None]):
                        for s in (STAGES if STAGES != [None] else [None]):
                            for gm in (GM if GM != [None] else [None]):
                                cfg = {}
                                if bm is not None: cfg["BLOCK_M"] = bm
                                if bn is not None: cfg["BLOCK_N"] = bn
                                if bk is not None: cfg["BLOCK_K"] = bk
                                if w is not None: cfg["num_warps"] = w
                                if s is not None: cfg["num_stages"] = s
                                if gm is not None: cfg["GROUP_M"] = gm
                                if cfg: grid.append(cfg)
    elif "BLOCK_SIZE" in names:
        BS = [64, 128, 256, 512, 1024]
        WARPS = [1, 2, 4, 8] if "num_warps" in names else [None]
        STAGES = [1, 2, 3, 4] if "num_stages" in names else [None]
        for bs in BS:
            for w in (WARPS if WARPS != [None] else [None]):
                for s in (STAGES if STAGES != [None] else [None]):
                    cfg = {"BLOCK_SIZE": bs}
                    if w is not None: cfg["num_warps"] = w
                    if s is not None: cfg["num_stages"] = s
                    grid.append(cfg)
    else:
        WARPS = [1, 2, 4, 8] if "num_warps" in names else []
        STAGES = [1, 2, 3, 4] if "num_stages" in names else []
        if WARPS or STAGES:
            for w in (WARPS or [None]):
                for s in (STAGES or [None]):
                    cfg = {}
                    if w is not None: cfg["num_warps"] = w
                    if s is not None: cfg["num_stages"] = s
                    grid.append(cfg)
    return grid


def find_triton_entrypoint(triton_mod: types.ModuleType):
    for name in ["run", "bench", "main", "launch"]:
        fn = getattr(triton_mod, name, None)
        if callable(fn):
            return fn, name
    for name, obj in vars(triton_mod).items():
        if callable(obj) and not isinstance(obj, type):
            try:
                if len(inspect.signature(obj).parameters) >= 1:
                    return obj, name
            except Exception:
                continue
    return None, None

def find_kernel_entrypoints(kernel_info: Tuple[str, List[int]], triton_mod: types.ModuleType) -> Dict[str, Any]:
    entries = {}
    for kernel_name, grid_args in kernel_info:
        log_debug(f"Kernel: {kernel_name}, Grid args: {grid_args}")
        obj = getattr(triton_mod, kernel_name, None)
        if callable(obj) and not isinstance(obj, type):
            entries[kernel_name] = obj
            log_info(f"Found kernel entrypoint: {kernel_name} with grid args {grid_args}")
    return entries

def device_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def time_callable(fn, warmup=5, iters=20) -> float:
    times = []
    for _ in range(warmup):
        _ = fn(); device_sync()
    for _ in range(iters):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); _ = fn(); end.record(); end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter(); _ = fn(); device_sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times)//2]


def filter_kwargs_for_signature(kwargs: Dict[str, Any], fn) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return {}


def tune_one(entry, entry_name: str, inputs: List[torch.Tensor], candidate_cfgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_ms, best_cfg, errors, attempts = float("inf"), None, 0, 0
    for meta in candidate_cfgs:
        meta_pass = filter_kwargs_for_signature(meta, entry)
        attempts += 1
        try:
            fn = lambda: entry(*inputs, **meta_pass)
            ms = time_callable(fn, warmup=3, iters=10)
        except Exception:
            errors += 1; continue
        if ms < best_ms:
            best_ms, best_cfg = ms, meta_pass
    return {"triton_entrypoint": entry_name, "attempted": attempts, "errors": errors,
            "best_latency_ms": best_ms if best_cfg else None, "best_config": best_cfg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path.cwd()))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="auto_tune.log", help="Log file path")
    args = parser.parse_args()

    # Initialize logging
    global logger
    logger = setup_logging(args.log_file)
    logger.info("Starting auto-tuning process")

    root = Path(args.root)
    py_dir, tri_dir = root / "pytorch_modules", root / "triton_modules"
    out_dir = root / "tuning_results"; out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")

    summary = []

    for py_file in sorted(py_dir.glob("*.py")):
        mod_name = py_file.stem
        tri_file = tri_dir / f"{mod_name}_triton.py"
        logger.info(f"\n=== Module: {mod_name} ===")

        try:
            # _install_triton_grid_shim()
            py_mod = dynamic_import_from_path(f"py_{mod_name}", str(py_file))
            inputs = get_python_inputs(py_mod, device)
        except Exception as e:
            logger.warning(f"[SKIP] Failed to import/get inputs for {mod_name}: {e}")
            continue

        cfgs_from_autotune: List[Dict[str, Any]] = []
        grid_dim: Optional[int] = None
        meta_names: List[str] = []

        if tri_file.exists():
            src = read_text(tri_file)
            kernel_info = detect_grid_dim_from_source(src)
            meta_names = extract_meta_param_names(src)
            logger.info(f"  -> Detected meta-parameters: {meta_names}")
        else:
            logger.warning(f"  [TUNE-SKIP] Triton file missing: {tri_file.name}")
            continue
        logger.info(f"  -> Detected kernel launches: {kernel_info}")
        logger.info(f"  -> Extracted meta-parameter names: {meta_names}")

        if kernel_info and kernel_info == []:
            logger.warning("Didn't detect triton kernel launches in source")

        tune_result = {"error": "triton_unavailable"} if triton is None else {"skipped": True}
        if tri_file.exists() and triton is not None:
            try:
                tri_mod = dynamic_import_from_path(f"tri_{mod_name}", str(tri_file))
                find_kernel_entrypoints(kernel_info, tri_mod)
                entry, entry_name = find_triton_entrypoint(tri_mod)
                logger.info(f"  -> Detected triton entrypoint: {entry_name}")
                # if entry is None:
                #     logger.warning("  [TUNE-SKIP] No callable entrypoint found")
                #     tune_result = {"error": "no_entrypoint"}
                # else:
                #     tune_result = tune_one(entry, entry_name, inputs, candidate_cfgs)
            except Exception as e:
                logger.error(f"  [TUNE-ERROR] {e}")
                tune_result = {"error": str(e)}
                sys.exit(1)
        elif tri_file.exists() and triton is None:
            logger.warning(f"  [TUNE-SKIP] Triton not available: {triton_import_error}")

        result = {
            "module": mod_name,
            "grid_dim": grid_dim,
            "used_configs_from_autotune": bool(cfgs_from_autotune),
            "num_candidate_configs": len(candidate_cfgs),
            "tuning": tune_result,
        }
        (out_dir / f"{mod_name}.json").write_text(json.dumps(result, indent=2))

        summary.append({
            "module": mod_name,
            "grid_dim": grid_dim,
            "candidates": len(candidate_cfgs),
            "best_ms": tune_result.get("best_latency_ms"),
            "best_cfg": tune_result.get("best_config"),
            "errors": tune_result.get("errors"),
        })

    import pandas as pd
    df = pd.DataFrame(summary)
    df_path = root / "autotune_summary.csv"
    df.to_csv(df_path, index=False)
    logger.info(f"\nWrote summary CSV: {df_path}")
    logger.info(f"JSON results in: {out_dir}")


if __name__ == "__main__":
    main()
