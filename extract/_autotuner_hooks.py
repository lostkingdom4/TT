"""Shared CachingAutotuner.run hooks for Inductor-emitted Triton kernels.

Used by the KernelBench driver to simultaneously (a) time each kernel launch
and (b) record tensor/scalar args, grids, and configs for downstream TTGIR
matching. The existing extract/run_inductor.py and extract/measure_kernel_latency.py
each use their own narrower hook; this helper combines both so we only need a
single forward pass to screen and capture.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List


@contextmanager
def combined_autotuner_hook(output_dir: Path | None = None):
    """Install a combined timing + shape-capture hook on CachingAutotuner.run.

    Yields a dict with two fields populated during the forward pass:
      - latencies: {kernel_name: [ms, ms, ...]}  (one entry per launch)
      - captured: {kernel_name: [shape_record, ...]}  (deduped by signature)

    If ``output_dir`` is given, per-input Triton source snapshots are saved
    there as ``{kname}_input{i}.py`` (same convention as run_inductor.py).
    """
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    latencies: Dict[str, List[float]] = defaultdict(list)
    captured: Dict[str, List[dict]] = {}
    seen_sigs: Dict[str, set] = {}

    original_run = CachingAutotuner.run

    def hooked_run(self, *args, grid, stream, benchmark_run=False, **kwargs):
        kname = getattr(getattr(self, "fn", None), "__name__", "unknown")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_run(
            self, *args, grid=grid, stream=stream,
            benchmark_run=benchmark_run, **kwargs,
        )
        end.record()
        torch.cuda.synchronize()
        latencies[kname].append(start.elapsed_time(end))

        try:
            tensor_args, scalar_args = [], []
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    tensor_args.append({
                        "arg_index": idx,
                        "shape": list(arg.shape),
                        "dtype": str(arg.dtype),
                        "device": str(arg.device),
                        "numel": int(arg.numel()),
                        "stride": list(arg.stride()),
                        "is_contiguous": bool(arg.is_contiguous()),
                    })
                elif isinstance(arg, (int, float, bool)):
                    scalar_args.append({
                        "arg_index": idx,
                        "value": arg,
                        "type": type(arg).__name__,
                    })

            config_kwargs: Dict[str, Any] = {}
            if hasattr(self, "launchers") and self.launchers:
                cfg = self.launchers[0].config
                config_kwargs = {
                    k: {"type": "scalar", "value": v,
                        "value_type": type(v).__name__}
                    for k, v in cfg.kwargs.items()
                }
                config_kwargs["num_warps"] = {
                    "type": "scalar", "value": cfg.num_warps, "value_type": "int"}
                config_kwargs["num_stages"] = {
                    "type": "scalar", "value": cfg.num_stages, "value_type": "int"}

            grid_val = grid
            if callable(grid):
                try:
                    launcher_cfg = {}
                    if hasattr(self, "launchers") and self.launchers:
                        launcher_cfg = dict(self.launchers[0].config.kwargs)
                    grid_val = grid(launcher_cfg)
                except Exception:
                    grid_val = str(grid)

            sig_parts = [f"{t['arg_index']}:{t['shape']}:{t['dtype']}"
                         for t in tensor_args]
            sig_parts += [f"{s['arg_index']}:{s['value']}" for s in scalar_args]
            sig = "|".join(sig_parts)

            if kname not in seen_sigs:
                seen_sigs[kname] = set()
                captured[kname] = []

            if sig not in seen_sigs[kname]:
                seen_sigs[kname].add(sig)
                if isinstance(grid_val, (tuple, list)):
                    grid_out = list(grid_val)
                else:
                    grid_out = str(grid_val)
                captured[kname].append({
                    "kernel": kname,
                    "tensor_args": tensor_args,
                    "scalar_args": scalar_args,
                    "kwargs": config_kwargs,
                    "grid": grid_out,
                    "total_args": len(args),
                    "input_index": len(captured[kname]),
                })

                if output_dir is not None:
                    src_file = getattr(self, "filename", None)
                    if src_file and Path(src_file).exists():
                        output_dir.mkdir(parents=True, exist_ok=True)
                        idx = len(captured[kname]) - 1
                        try:
                            shutil.copy2(
                                src_file,
                                str(output_dir / f"{kname}_input{idx}.py"),
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        return result

    CachingAutotuner.run = hooked_run
    state = {"latencies": latencies, "captured": captured}
    try:
        yield state
    finally:
        CachingAutotuner.run = original_run
