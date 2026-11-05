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


def detect_grid_dim_from_source(src: str) -> Optional[Tuple[str, List[int]]]:
    """
        Infer grid dimensionality from Triton kernel launches.
        Returns:
        (kernel_name, [grid args as strings])
    Example:
        'triton_poi_fused_add_0[grid(256)](...)' -> ("triton_poi_fused_add_0", ["256"])
        'foo[grid(64, 32)](...)' -> ("foo", ["64", "32"])
    """
    outputs = []
    # --- Case 1: kernel[grid(...)]( ... )
    pattern_launch = re.compile(
        r'^\s*([A-Za-z_]\w*)\s*\[\s*grid\s*\((.*?)\)\s*\]\s*(?:\(|$)',
        re.MULTILINE | re.DOTALL
    )
    for kernel_name, grid_inside in pattern_launch.findall(src):
        part = [p.strip() for p in grid_inside.split(",") if p.strip()]
        if part:
            outputs.append((kernel_name, [int(p) for p in part]))
        else:
            print(f"Warning: Could not parse grid args in launch: {grid_inside}")
            sys.exit(1)
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

def find_kernel_entrypoints(kernel_info: List[Dict[str, Any]], triton_mod: types.ModuleType) -> Dict[str, Any]:
    entries = {}
    for kernel_name, grid_args in kernel_info:
        obj = getattr(triton_mod, kernel_name, None)
        if callable(obj) and not isinstance(obj, type):
            entries[kernel_name] = obj
            print(f"Found kernel entrypoint: {kernel_name} with grid args {grid_args}")
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
    args = parser.parse_args()

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
        print(f"\n=== Module: {mod_name} ===")

        try:
            _install_triton_grid_shim()
            py_mod = dynamic_import_from_path(f"py_{mod_name}", str(py_file))
            inputs = get_python_inputs(py_mod, device)
        except Exception as e:
            print(f"[SKIP] Failed to import/get inputs for {mod_name}: {e}")
            continue
        
        print(py_mod)
        print(inputs)

        cfgs_from_autotune: List[Dict[str, Any]] = []
        grid_dim: Optional[int] = None
        meta_names: List[str] = []

        if tri_file.exists():
            src = read_text(tri_file)
            print(src)
            kernel_info = detect_grid_dim_from_source(src)
            meta_names = extract_meta_param_names(src)
            print(f"  -> Detected meta-parameters: {meta_names}")
        else:
            print(f"  [TUNE-SKIP] Triton file missing: {tri_file.name}")
            
        if kernel_info and kernel_info == []:
            print("Didn't detect triton kernel launches in source")

        tune_result = {"error": "triton_unavailable"} if triton is None else {"skipped": True}
        if tri_file.exists() and triton is not None:
            try:
                tri_mod = dynamic_import_from_path(f"tri_{mod_name}", str(tri_file))
                find_kernel_entrypoints(kernel_info, tri_mod)
                exit()
                entry, entry_name = find_triton_entrypoint(tri_mod)
                print(f"  -> Detected triton entrypoint: {entry_name}")
                exit()
                # if entry is None:
                #     print("  [TUNE-SKIP] No callable entrypoint found")
                #     tune_result = {"error": "no_entrypoint"}
                # else:
                #     tune_result = tune_one(entry, entry_name, inputs, candidate_cfgs)
            except Exception as e:
                print(f"  [TUNE-ERROR] {e}")
                tune_result = {"error": str(e)}
                sys.exit(1)
        elif tri_file.exists() and triton is None:
            print(f"  [TUNE-SKIP] Triton not available: {triton_import_error}")

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
    print(f"\nWrote summary CSV: {df_path}")
    print(f"JSON results in: {out_dir}")


if __name__ == "__main__":
    main()
