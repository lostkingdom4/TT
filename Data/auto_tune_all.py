import torch
import argparse
import json
import sys
import shutil
from pathlib import Path
from importlib import util as importlib_util
import types
import os
import triton
import triton.language as tl
import time
from datetime import datetime
from tqdm import tqdm as tqmd
import logging

from tune_one_file import autotune_file

from json_accumulator import AutotuneResultsStore

def install_inductor_shims():
    """
    Provide minimal stand-ins for symbols some Inductor-generated Triton code imports.
    Works across PyTorch versions where these aren't exported.
    """
    mod_name = "torch._inductor.runtime.triton_heuristics"
    m = sys.modules.get(mod_name)
    if m is None:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m

    # grid(...) -> return a tuple (1D/2D/3D)
    if not hasattr(m, "grid"):
        def grid(*sizes, **kwargs):
            if sizes and callable(sizes[0]):  # e.g., grid(lambda META: ...)
                return (1,)
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                return tuple(sizes[0])
            return tuple(sizes)
        m.grid = grid

    # Minimal FixedGrid that acts like a tuple of dims
    if not hasattr(m, "FixedGrid"):
        class FixedGrid:
            def __init__(self, *sizes):
                # allow FixedGrid((x,y,z)) or FixedGrid(x,y,z)
                if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                    self.sizes = tuple(sizes[0])
                else:
                    self.sizes = tuple(sizes)
            # Behave tuple-like
            def __iter__(self):
                return iter(self.sizes)
            def __len__(self):
                return len(self.sizes)
            def __getitem__(self, i):
                return self.sizes[i]
            def __repr__(self):
                return f"FixedGrid{self.sizes}"
            # Some code may call it (rare); just return the tuple
            def __call__(self, *_, **__):
                return self.sizes
        m.FixedGrid = FixedGrid

def delete_all_contents(dir_path):
    p = Path(dir_path)
    if not p.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")
    for item in p.iterdir():
        if item.is_dir():
            shutil.rmtree(item)    # remove directory recursively
        else:
            item.unlink()          # remove file/symlink

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path.cwd()))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file (default: <root>/auto_tune_all.log)")
    parser.add_argument('--start-index', type=int, default=0, help='Start index for processing files (default: 0)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of repos to process')
    parser.add_argument("--timeout-per-config", type=int, default=20, help="Maximum time in seconds to spend on each config within a kernel (default: 60)")
    parser.add_argument("--log-level", type=str,
                       default=os.getenv("TUNE_LOG_LEVEL", "INFO"),
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Console log level (default: INFO, or TUNE_LOG_LEVEL env var). File always logs DEBUG+")
    args = parser.parse_args()

    results_store = AutotuneResultsStore("triton_autotune_results.json")

    root = Path(args.root)
    py_dir, tri_dir = root / "pytorch_modules", root / "triton_modules"
    out_dir = root / "tuning_results"; out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "triton_cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    # install_inductor_shims()

    # --- Setup logging ---
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        # Generate log filename with timestamp and range info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        range_str = f"start{args.start_index}"
        if args.limit:
            range_str += f"_limit{args.limit}"
        log_filename = f"auto_tune_all_{timestamp}_{range_str}.log"
        log_path = root / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger to capture all logging from submodules
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_log_level = getattr(logging, args.log_level.upper())
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Get a logger for this module
    logger = logging.getLogger(__name__)

    logger.info("===== auto_tune_all run started =====")
    logger.info(f"Logging to: {log_path}")

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
    error = 0

    # Get sorted list of files and apply start_index and limit
    all_files = sorted(py_dir.glob("*.py"))
    end_index = args.start_index + args.limit if args.limit else None
    files_to_process = all_files[args.start_index:end_index]

    logger.info(f"Processing files {args.start_index} to {end_index if end_index else len(all_files)} (total: {len(files_to_process)})")

    for py_file in tqmd(files_to_process, desc="Modules", ncols=80):
        mod_name = py_file.stem
        tri_file = tri_dir / f"{mod_name}_triton.py"
        logger.info(f"=== Module: {mod_name} ===")

        if tri_file.exists():
            delete_all_contents(cache_dir)
            try:
                rs = autotune_file(tri_file, timeout_per_config=args.timeout_per_config)
                results_store.add_results(mod_name, str(tri_file), rs)
            except Exception as e:
                logger.error(f"Error autotuning {mod_name}: {e}")
                error += 1
                continue

    #     for r in rs:
    #         grid_dim = r.get("grid_dim")
    #         candidate_cfgs = r.get("candidate_cfgs", [])
    #         tune_result = r.get("tune_result", {})
    #         with open(out_dir / f"{mod_name}_grid{grid_dim}.json", "w") as f:
    #             json.dump(r, f, indent=2)
    #         print(f"Wrote: {out_dir / f'{mod_name}_grid{grid_dim}.json'}")
    #     summary.append({
    #         "module": mod_name,
    #         "grid_dim": grid_dim,
    #         "candidates": len(candidate_cfgs),
    #         "best_ms": tune_result.get("best_latency_ms"),
    #         "best_cfg": tune_result.get("best_config"),
    #         "errors": tune_result.get("errors"),
    #     })

    # import pandas as pd
    # df = pd.DataFrame(summary)
    # df_path = root / "autotune_summary.csv"
    # df.to_csv(df_path, index=False)
    # print(f"\nWrote summary CSV: {df_path}")
    # print(f"JSON results in: {out_dir}")

    logger.info("===== auto_tune_all run completed =====")

    # Cleanup logging handlers
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
if __name__ == "__main__":
    install_inductor_shims()
    main()