#!/usr/bin/env python3
# extract_function_from_data.py
# Scan a directory, load each *.py by absolute path, call get_inputs(), save tensors, log summary.

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Optional: PyTorch is only needed if your get_inputs() returns tensors
try:
    import torch
except Exception:
    torch = None  # We’ll guard uses


def load_module_from_path(py_path: Path, unique_name: str):
    """Import a python file as a module using its file path (no PYTHONPATH required)."""
    spec = importlib.util.spec_from_file_location(unique_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    # Insert to sys.modules so relative imports in the file have a better chance to work
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def is_tensor(x: Any) -> bool:
    return (torch is not None) and isinstance(x, torch.Tensor)


def tensor_meta(t: "torch.Tensor") -> Dict[str, Any]:
    d = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    try:
        # Don’t move device unless needed; just for stats we can do a lightweight cast
        d["mean"] = float(t.detach().float().mean().item())
        d["std"] = float(t.detach().float().std().item())
    except Exception:
        pass
    return d


def safe_meta(x: Any) -> Dict[str, Any]:
    if is_tensor(x):
        return tensor_meta(x)
    # Extend here if you have other container types you want to support
    return {"type": str(type(x))}


def record_one_file(py_path: Path, idx: int, out_dir: Path, seed: int = 2025) -> Dict[str, Any]:
    """Load a .py file, call get_inputs() if present, save returned tensors, return a summary entry."""
    unique_name = f"kbmod_{idx}_{py_path.stem}"
    entry: Dict[str, Any] = {
        "file": str(py_path),
        "module_name": unique_name,
        "status": "unknown",
        "saved": [],
        "meta": [],
        "num_items": 0,
    }

    try:
        mod = load_module_from_path(py_path, unique_name)
    except Exception as e:
        entry["status"] = "import_error"
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc(limit=3)
        return entry

    if not hasattr(mod, "get_inputs"):
        entry["status"] = "skip_no_get_inputs"
        return entry

    # Optional: make the run reproducible across files
    try:
        if torch is not None:
            torch.manual_seed(seed)
    except Exception:
        pass

    # Call get_inputs()
    try:
        ret = mod.get_inputs()
    except Exception as e:
        entry["status"] = "error_in_get_inputs"
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc(limit=3)
        return entry

    # Normalize to list
    if not isinstance(ret, (list, tuple)):
        ret = [ret]
    entry["num_items"] = len(ret)

    # Save results
    for j, obj in enumerate(ret):
        entry["meta"].append(safe_meta(obj))
        if is_tensor(obj):
            out_file = out_dir / f"{py_path.stem}__{idx}_{j}.pt"
            try:
                # Save a CPU copy to make files portable
                # torch.save(obj.detach().cpu(), out_file)
                entry["saved"].append(str(out_file))
            except Exception as e:
                entry["saved"].append(f"<save_failed: {type(e).__name__}: {e}>")
        else:
            entry["saved"].append("<not_a_tensor>")

    entry["status"] = "ok"
    return entry


def main():
    parser = argparse.ArgumentParser(description="Record all return values from get_inputs() across a directory.")
    parser.add_argument("--dir", type=str, default="Data/pytorch_modules/", help="Directory to scan recursively for *.py files")
    parser.add_argument("--out", type=str, default="./recorded_inputs", help="Output directory for tensors + summary.json")
    parser.add_argument("--exclude", type=str, nargs="*", default=[], help="Filename patterns to exclude (substr match)")
    parser.add_argument("--seed", type=int, default=2025, help="Seed for reproducibility before calling get_inputs()")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    this_file = Path(__file__).resolve()
    py_files: List[Path] = []
    for p in sorted(target_dir.rglob("*.py")):
        rp = p.resolve()
        if rp == this_file:
            continue  # skip this script itself
        name = rp.name
        if any(ex in name for ex in args.exclude):
            continue
        if "__init__" in name:
            continue
        py_files.append(rp)

    print(f"Found {len(py_files)} python files under {target_dir}")

    summary: List[Dict[str, Any]] = []
    for i, py in enumerate(py_files):
        res = record_one_file(py, i, out_dir, seed=args.seed)
        status = res.get("status", "unknown")
        print(f"[{i:03d}] {py.name}: {status}")
        summary.append(res)

    # Write JSON summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    ok_count = sum(1 for s in summary if s.get("status") == "ok")
    print(f"\nDone. ok={ok_count}/{len(summary)}  → tensors: {out_dir}  → summary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    # Ensure project root is on sys.path for any relative imports inside scanned files
    sys.path.insert(0, str(Path(".").resolve()))
    main()
