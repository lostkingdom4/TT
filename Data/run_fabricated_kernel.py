import json
import time
from pathlib import Path
from typing import Optional, List
import torch
import importlib.util

# ---------- utilities ----------
def _import_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()

def _grid_1d(xnumel: int, XBLOCK: int):
    return ((xnumel + XBLOCK - 1) // XBLOCK,)

# ---------- main: run a kernel with fabricated inputs ----------
def run_fabricated_kernel(
    triton_py_file: str,
    kernel_name: str,
    inputs_dir: str,
    *,
    XBLOCK: int = 256,
    num_warps: int = 4,
    num_stages: int = 1,
    warmup: int = 25,
    iters: int = 100,
    device: Optional[torch.device] = None,
) -> float:
    """
    Load tensors from fabricate_inputs_for_file(), import the Triton kernel, and run it.

    Expects in `inputs_dir`:
      - {kernel_name}_meta.json  -> {"xnumel": int, "param_names": [...], "xnumel_param_index": int}
      - {kernel_name}_args.pt    -> [tensor0, tensor1, ...]  # one per pointer-like arg before xnumel

    Returns:
      Median latency in milliseconds over `iters` runs.
    """
    inputs_dir = Path(inputs_dir)
    meta_path = inputs_dir / f"{kernel_name}_meta.json"
    args_path = inputs_dir / f"{kernel_name}_args.pt"

    if not meta_path.is_file() or not args_path.is_file():
        raise FileNotFoundError(f"Missing artifacts for {kernel_name} in {inputs_dir}. "
                                f"Expected {meta_path.name} and {args_path.name}")

    meta = json.loads(meta_path.read_text())
    xnumel = int(meta["xnumel"])
    param_names: List[str] = meta["param_names"]

    # Import the Triton module and fetch the kernel
    mod = _import_from_path(f"mod_{Path(triton_py_file).stem}", triton_py_file)
    if not hasattr(mod, kernel_name):
        raise AttributeError(f"Kernel {kernel_name} not found in {triton_py_file}")
    kernel_fn = getattr(mod, kernel_name)

    # Load fabricated flat buffers (already correct length)
    flat_args: List[torch.Tensor] = torch.load(args_path)

    dev = device or _device()
    flat_args = [t.to(dev, non_blocking=True) for t in flat_args]

    # Build launch grid and full positional args list: (*buffers, xnumel)
    grid = _grid_1d(xnumel, XBLOCK)
    pos_args = [*flat_args, xnumel]

    # Warmup
    for _ in range(warmup):
        kernel_fn[grid](*pos_args, XBLOCK=XBLOCK, num_warps=num_warps, num_stages=num_stages)
        _sync()

    # Timed runs
    times_ms: List[float] = []
    use_cuda_timer = torch.cuda.is_available()
    if use_cuda_timer:
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel_fn[grid](*pos_args, XBLOCK=XBLOCK, num_warps=num_warps, num_stages=num_stages)
            end.record(); end.synchronize()
            times_ms.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            kernel_fn[grid](*pos_args, XBLOCK=XBLOCK, num_warps=num_warps, num_stages=num_stages)
            _sync()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1e3)

    times_ms.sort()
    med = times_ms[len(times_ms)//2]
    print(f"[{kernel_name}] xnumel={xnumel}  grid={grid}  "
          f"XBLOCK={XBLOCK} warps={num_warps} stages={num_stages}  "
          f"median={med:.4f} ms")
    return med


run_fabricated_kernel(
    triton_py_file="./triton_modules/_ASPP_triton.py",
    kernel_name="triton_poi_fused_add_convolution_0",
    inputs_dir=".",
    XBLOCK=256, num_warps=4, num_stages=1
)