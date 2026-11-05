#!/usr/bin/env python3
import os, time, math
os.environ.setdefault("TRITON_CACHE_DIR", "./triton_cache")
# export TRITON_PRINT_AUTOTUNING=1
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

import torch
import triton
import triton.language as tl

# ---------- device helpers ----------
def has_xpu(): return hasattr(torch, "xpu") and torch.xpu.is_available()
def sync():
    if has_xpu(): torch.xpu.synchronize()
    
def remove_cache():
    import shutil
    shutil.rmtree(os.environ["TRITON_CACHE_DIR"], ignore_errors=True)

device = torch.device("xpu" if has_xpu() else "cpu")

# ---------- custom do_bench passed to triton.autotune ----------
def xpu_do_bench(fn, quantiles=(0.5,)):
    """
    fn: a zero-arg callable that launches the kernel once.
    quantiles: ignored except we return the median (0.5).
    Returns: median latency in *milliseconds*.
    """
    # warmup
    for _ in range(25):
        fn(); sync()
    # measure
    iters = 100
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(); sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
    times_ms.sort()
    # return median by default
    return times_ms[len(times_ms)//2]


block_size = [32, 64, 128, 256, 512, 1024, 2048, 4096]
num_warps = [1, 2, 4, 8, 16]
num_stages = [1, 2, 3, 4]

# ---------- kernel + autotune ----------
CONFIGS = []
for b in block_size:
    for w in num_warps:
        for s in num_stages:
            if b * w <= 4096:   # heuristic to avoid OOM
                CONFIGS.append(triton.Config({"BLOCK_SIZE": b}, num_warps=w, num_stages=s))

@triton.autotune(
    configs=CONFIGS,
    key=["N"],                     # re-tune when N changes
    do_bench=xpu_do_bench,         # <— your custom timer
    # Optional: if your kernel accumulates into outputs, prevent double updates:
    reset_to_zero=['y'],          # use the exact arg name in the signature below
    cache_results = True,        # cache results based on key
)
@triton.jit
def softmax_rowwise_kernel(x, y, M, N, sxm, sxn, sym, syn, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x_row = x + row * sxm + offs * sxn
    y_row = y + row * sym + offs * syn
    vals = tl.load(x_row, mask=mask, other=-float("inf"))
    vals = vals - tl.max(vals, axis=0)
    num = tl.exp(vals)
    den = tl.sum(num, axis=0)
    tl.store(y_row, num / den, mask=mask)

def softmax_rowwise(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.device.type == device.type
    M, N = x.shape
    y = torch.empty_like(x)
    # one program per row; tile must cover the row
    softmax_rowwise_kernel[(M,)](
        x, y, M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        # BLOCK_SIZE=BLOCK
    )
    return y

# ---------- demo ----------
if __name__ == "__main__":
    if device.type != "xpu":
        print("Warning: XPU not available; running on CPU for correctness only.")
    remove_cache()
    torch.manual_seed(0)
    M, N = 4096, 2048
    x = torch.randn(M, N, device=device, dtype=torch.float32)

    # Trigger JIT + autotune (first call not timed)
    y = softmax_rowwise(x); sync()

    # Show result correctness
    ref = torch.softmax(x, dim=1)
    print("max error:", (y - ref).abs().max().item())

    # If you also want high-level timing afterwards, you can reuse the same bench:
    def run_once(): softmax_rowwise(x)
    print("Median latency (ms):", xpu_do_bench(run_once))
