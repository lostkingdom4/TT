import re
import sys
import types
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging

import torch
import ast
import os, io, re
import contextlib
import json

import triton
import triton.language as tl
import triton.testing
import multiprocessing
import time
import queue
import gc


from process_cache import search_for_file, config, get_best_config

os.environ.setdefault("TRITON_CACHE_DIR", "./triton_cache")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

def has_xpu(): return hasattr(torch, "xpu") and torch.xpu.is_available()
def sync():
    if has_xpu(): torch.xpu.synchronize()

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

# ---------------------------
# 0) Make 'grid' import safe
# ---------------------------
def _install_triton_grid_shim():
    """
    Inductor code often does:
      from torch._inductor.runtime.triton_heuristics import grid
    Some PyTorch builds don't export it. Provide a tiny shim.
    """
    mod_name = 'torch._inductor.runtime.triton_heuristics'
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        def grid(*sizes, **kwargs):
            if sizes and callable(sizes[0]):
                # if someone passed a lambda META: ... ; return harmless default
                return (1,)
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                return tuple(sizes[0])
            return tuple(sizes)
        m.grid = grid
        sys.modules[mod_name] = m

_install_triton_grid_shim()

# ---------------------------
# 1) Dynamic import helpers
# ---------------------------
def import_from_path(name: str, path: str):
    _install_triton_grid_shim()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ---------------------------
# 2) Source parsing: xnumel
# ---------------------------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

# def infer_xnumel_for_kernel(src: str, kernel_name: str) -> Optional[int]:
#     """
#     Look for 'def kernel(...):' followed by 'xnumel = <int>' inside body.
#     Works well for Inductor-generated 1D kernels.
#     """
#     # Find the function block for this kernel
#     # crude but effective: grab from 'def name' to next 'def ' or EOF
#     pat = rf"@triton\.jit\s*def\s+{re.escape(kernel_name)}\s*\(.*?\):(.+?)(?=\n@triton\.jit|\ndef\s|\Z)"
#     m = re.search(pat, src, flags=re.DOTALL)
#     if not m:
#         return None
#     body = m.group(1)
#     m2 = re.search(r"\bxnumel\s*=\s*(\d+)", body)
#     return int(m2.group(1)) if m2 else None

def infer_numel_for_kernel(src: str, kernel_name: str) -> Tuple[Optional[List[str]], Optional[List[int]], Optional[List[str]]]:
    """
    Find the Triton kernel by name, capture its parameter list (inside the
    parentheses) and its body, then:
      - parse and return the parameter names
      - look for parameters ending with 'numel' and extract their values from body or call sites
      - return separate lists for numel param names, their values, and other param names

    Returns: (numel_param_names, numel_values, other_param_names)
    """
    # Capture both the params and the body with named groups
    pat = rf"@triton\.jit\s*def\s+{re.escape(kernel_name)}\s*\((?P<params>.*?)\)\s*:(?P<body>.+?)(?=\n@triton\.jit|\ndef\s|\Z)"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m:
        return None, None, None

    params_src = m.group("params").strip()
    body = m.group("body")

    # Robustly parse parameter *names* via AST (handles annotations/defaults)
    all_param_names: Optional[List[str]] = None
    try:
        dummy = f"def __tmp__({params_src}):\n    pass\n"
        mod = ast.parse(dummy)
        f = mod.body[0]  # type: ignore[index]
        # Ensure f is a FunctionDef before accessing args
        if isinstance(f, ast.FunctionDef):
            # Collect positionals; add others if you care (vararg/kwonly/kwarg)
            posonly = getattr(f.args, "posonlyargs", [])
            args = f.args.args
            all_param_names = [a.arg for a in (*posonly, *args)]
    except Exception:
        # Fallback: naive split (good enough for simple signatures)
        if params_src:
            all_param_names = [p.split(":", 1)[0].split("=", 1)[0].strip()
                              for p in params_src.split(",") if p.strip()]

    if not all_param_names:
        return None, None, None

    # Separate numel parameters from other parameters
    numel_param_names = [name for name in all_param_names if name.endswith("numel")]
    other_param_names = [name for name in all_param_names if not name.endswith("numel")]

    # Extract values for all numel parameters
    numel_values = []

    for numel_param in numel_param_names:
        # First try to find literal assignment in body
        pattern = rf"\b{re.escape(numel_param)}\s*=\s*(\d+)"
        m_literal = re.search(pattern, body)

        if m_literal:
            value = int(m_literal.group(1))
            log.debug(f"Found {numel_param}={value} in kernel body")
            numel_values.append(value)
        else:
            # Try to extract from call sites
            value = _extract_numel_from_call_sites(src, kernel_name, numel_param, all_param_names)

            if value is None:
                # Try to infer from other patterns in the body if this is xnumel
                if numel_param == "xnumel":
                    other_numel = re.search(r"\b(y|z|r0_)?numel\s*=\s*(\d+)", body)
                    if other_numel:
                        value = int(other_numel.group(2))
                        log.debug(f"Found {other_numel.group(1) or ''}numel={value}, using this as {numel_param}")
                    else:
                        value = 16384  # Fallback for xnumel
                        log.debug(f"Using fallback {numel_param}={value} (common Inductor size)")
                else:
                    # For other numel parameters, use reasonable defaults
                    if numel_param == "ynumel":
                        value = 16384  # Default for 2D kernels
                    elif numel_param.startswith("r") and numel_param.endswith("_numel"):
                        value = 64  # Default for reduction dimensions
                    else:
                        value = 1024  # General fallback
                    log.debug(f"Using fallback {numel_param}={value}")

            numel_values.append(value)

    log.debug(f"Extracted numel parameters: {dict(zip(numel_param_names, numel_values))}")
    log.debug(f"Other parameters: {other_param_names}")

    return numel_param_names, numel_values, other_param_names

def _extract_numel_from_call_sites(src: str, kernel_name: str, numel_param: str, param_names: List[str]) -> Optional[int]:
    """
    Scan the source code for kernel invocation sites to extract actual numel values.
    Looks for patterns like: kernel_name[grid](*args, numel_value, ...)
    """
    # Find kernel invocation patterns
    # Pattern: kernel_name[grid_size, ...](arg1, arg2, numel_value, ...)
    call_pattern = rf"{re.escape(kernel_name)}\[[^\]]+\]\s*\([^)]+\)"

    call_matches = re.finditer(call_pattern, src)

    for call_match in call_matches:
        call_text = call_match.group(0)
        log.debug(f"Found kernel call: {call_text[:100]}...")

        # Extract arguments from the call
        # Find the argument list inside parentheses
        args_match = re.search(r'\([^)]+\)', call_text)
        if not args_match:
            continue

        args_text = args_match.group(0)[1:-1]  # Remove parentheses

        # Split arguments, being careful about nested calls
        args = _split_function_args(args_text)

        # Try to find numel parameter based on parameter position
        try:
            if numel_param in param_names:
                numel_idx = param_names.index(numel_param)
                if numel_idx < len(args):
                    arg_value = args[numel_idx].strip()
                    # Try to extract numeric value
                    if arg_value.isdigit():
                        extracted_numel = int(arg_value)
                        log.debug(f"Extracted {numel_param}={extracted_numel} from call site")
                        return extracted_numel
                    else:
                        log.debug(f"Found {numel_param} argument '{arg_value}' but it's not a simple integer")
        except (ValueError, IndexError):
            continue

    return None

def _extract_xnumel_from_call_sites(src: str, kernel_name: str, param_names: Optional[List[str]]) -> Optional[int]:
    """
    Legacy function - wraps the new _extract_numel_from_call_sites for backward compatibility
    """
    if not param_names:
        return None
    return _extract_numel_from_call_sites(src, kernel_name, "xnumel", param_names)

def _split_function_args(args_text: str) -> List[str]:
    """
    Split function arguments, handling nested parentheses and brackets.
    """
    args = []
    current_arg = ""
    paren_depth = 0
    bracket_depth = 0

    for char in args_text:
        if char == ',' and paren_depth == 0 and bracket_depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            current_arg += char

    if current_arg.strip():
        args.append(current_arg.strip())

    return args

from pathlib import Path
from typing import List
import importlib.util
import sys
import torch


def _import_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Could not create spec for {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    if mod is None:
        raise ImportError(f"Could not create module for {name}")
    sys.modules[name] = mod
    if spec.loader is None:
        raise ImportError(f"Spec has no loader for {name}")
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def get_inputs(py_path: str) -> List[torch.Tensor]:
    p = Path(py_path)
    fname = p.name  # e.g., A2Block_triton.py
    if not fname.endswith("_triton.py"):
        raise ValueError(f"Expected '*_triton.py' file, got: {fname}")

    op_name = fname[:-len("_triton.py")]  # "A2Block"
    # Data directory is one level above triton_modules
    data_dir = p.parent.parent
    orig_path = data_dir / "pytorch_modules" / f"{op_name}.py"
    if not orig_path.is_file():
        raise FileNotFoundError(f"Original PyTorch file not found: {orig_path}")

    log.debug(f"Loading original module from: {orig_path}")
    mod = _import_from_path(f"pytorch_{op_name}", str(orig_path))
    log.debug(f"Imported module: {mod}")

    # Defaults
    B = 8
    H = W = 56

    # Try to read inplane/plane from get_init_inputs()
    C = 4
    if hasattr(mod, "get_init_inputs"):
        try:
            init_list = mod.get_init_inputs()
            # Expected: [[], {'inplane': 4, 'plane': 4}]
            if (isinstance(init_list, list) and len(init_list) == 2
                    and isinstance(init_list[1], dict)):
                C = int(init_list[1].get("inplane", C))
        except Exception:
            pass

    # If the original used 4, bump to a meaningful size
    if C <= 4:
        C = 64

    return [torch.rand([B, C, H, W])]

# ---------------------------
# 3) Build dummy args
# ---------------------------
def make_dummy_args_for_kernel(numel_param_names: List[str], numel_values: List[int], other_param_names: List[str], device: torch.device):
    """
    Heuristic arg builder for Inductor Triton kernels.
    Handles both pointwise (triton_poi_) and reduction (triton_per_/triton_red_) kernels.

    Args:
        numel_param_names: List of parameter names ending with 'numel' (e.g., ['xnumel', 'ynumel'])
        numel_values: List of corresponding values for numel parameters
        other_param_names: List of non-numel parameter names (pointers, etc.)
        device: Target device for tensor allocation
    """
    args = []
    kw = {}

    # Combine all parameter names for processing
    param_names = numel_param_names + other_param_names
    log.debug(f"All param names: {param_names}")
    log.debug(f"Numel params: {dict(zip(numel_param_names, numel_values))}")

    # Find all non-constexpr parameters
    ptr_params = []
    scalar_params = []

    for name in param_names:
        if name in numel_param_names:
            scalar_params.append(name)
        elif "_ptr" in name:
            ptr_params.append(name)
        elif name.endswith("BLOCK") or name in ["num_warps", "num_stages"]:
            # These are constexpr, handled separately during kernel execution
            continue
        else:
            # Assume other parameters are pointers
            ptr_params.append(name)

    # Create a lookup dict for numel values
    numel_dict = dict(zip(numel_param_names, numel_values))

    # Get primary numel values for buffer size calculation
    xnumel = numel_dict.get("xnumel", max(numel_values) if numel_values else 16384)
    ynumel = numel_dict.get("ynumel", xnumel)  # Default for 2D kernels
    r0_numel = numel_dict.get("r0_numel", 64)  # Default reduction dimension

    # Calculate maximum possible buffer size needed
    max_buffer_size = max(numel_values) if numel_values else 16384

    # For 2D kernels, multiply dimensions
    if "ynumel" in numel_dict and "xnumel" in numel_dict:
        max_buffer_size = max(max_buffer_size, ynumel * xnumel)

    # For reduction kernels, account for reduction dimensions
    if any("r" in name and "numel" in name for name in numel_param_names):
        reduction_size = max([v for k, v in numel_dict.items() if "r" in k and "numel" in k], default=64)
        max_buffer_size = max(max_buffer_size, xnumel * reduction_size)

    # Smart buffer sizing - reduce safety margin for small problems to prevent OOM
    total_problem_size = sum(numel_values)
    if total_problem_size <= 8192:  # Small problems need less safety margin
        safety_factor = 2
    elif total_problem_size <= 65536:  # Medium problems
        safety_factor = 3
    else:  # Large problems
        safety_factor = 4

    safe_buffer_size = max_buffer_size * safety_factor
    min_buffer_size = 4096  # Minimum reasonable size
    buffer_size = max(safe_buffer_size, min_buffer_size)

    log.debug(f"Buffer sizing: problem_size={total_problem_size}, max_buffer_size={max_buffer_size}, safety_factor={safety_factor}, buffer_size={buffer_size}")

    # Allocate buffers for pointer parameters
    buffers = []
    for i, param_name in enumerate(ptr_params):
        # Use consistent large buffer size for all pointers to avoid memory access issues
        buf = torch.empty(buffer_size, dtype=torch.float32, device=device)
        # Initialize with small random values to avoid NaN/inf propagation
        torch.nn.init.uniform_(buf, -0.1, 0.1)
        buffers.append(buf)

    # Add buffers to args
    args.extend(buffers)

    # Add scalar parameters in order they appear, using the actual numel values
    for param_name in scalar_params:
        if param_name in numel_dict:
            args.append(numel_dict[param_name])
        else:
            # Fallback for unexpected scalar parameters
            args.append(max(1024, xnumel // 16))  # Scale with problem size

    for i, b in enumerate(buffers):
        log.debug(f"Buffer {i}: {b.shape} (size: {b.numel()})")

    log.debug(f"Scalar args: {[arg for arg in args if not isinstance(arg, torch.Tensor)]}")

    # meta-params (constexpr) will be passed as **kw later (XBLOCK, YBLOCK, R0_BLOCK, num_warps, num_stages)
    return args, kw# ---------------------------
# 4) Timing with timeout support
# ---------------------------
class TimeoutException(Exception):
    """Exception raised when a kernel execution times out"""
    pass

def with_timeout(func, timeout_seconds=30):
    """
    Execute a function with a timeout using signal alarm (Unix only).
    Works better with GPU operations than threading/multiprocessing approaches.

    Args:
        func: Function to execute
        timeout_seconds: Maximum time to wait in seconds

    Returns:
        Function result or raises TimeoutException
    """
    import signal
    import platform

    # Check if we're on a Unix-like system that supports signals
    if platform.system() == 'Windows':
        # Fallback to thread timeout on Windows
        return _thread_timeout(func, timeout_seconds)

    # Unix signal-based timeout
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Kernel execution timed out after {timeout_seconds} seconds")

    # Save old handler and set new one
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        # Set alarm
        signal.alarm(int(timeout_seconds))
        result = func()
        signal.alarm(0)  # Cancel alarm
        return result

    except TimeoutException:
        signal.alarm(0)  # Cancel alarm
        log.debug(f"    WARNING: Signal-based timeout after {timeout_seconds}s - attempting aggressive cleanup")

        # Aggressive GPU cleanup inspired by spawn test approach
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Clear cache to free any stuck kernels
                torch.cuda.empty_cache()
            elif has_xpu():
                torch.xpu.synchronize()
                if hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()

            # Force garbage collection
            gc.collect()

        except Exception as cleanup_error:
            log.debug(f"    Cleanup failed: {cleanup_error}")
        raise

    finally:
        # Restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def _thread_timeout(func, timeout_seconds):
    """Fallback thread-based timeout for Windows or when signals unavailable"""
    def target_func(result_container, exception_container):
        try:
            result = func()
            result_container[0] = result
        except Exception as e:
            exception_container[0] = e

    result_container = [None]
    exception_container = [None]

    thread = threading.Thread(target=target_func, args=(result_container, exception_container))
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        log.debug(f"    WARNING: Thread-based timeout after {timeout_seconds}s (thread still running)")
        # Aggressive GPU cleanup inspired by spawn test approach
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif has_xpu():
                torch.xpu.synchronize()
                if hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()
            # Force garbage collection
            gc.collect()
        except Exception as cleanup_error:
            log.debug(f"    Thread timeout cleanup failed: {cleanup_error}")
        raise TimeoutException(f"Kernel execution timed out after {timeout_seconds} seconds")

    if exception_container[0] is not None:
        raise exception_container[0]

    return result_container[0]

def device_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Optional: Clear cache periodically to avoid memory issues
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
        # XPU equivalent if available
        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()

def cleanup_memory():
    """Aggressive memory cleanup to prevent OOM during intensive tuning"""
    import gc
    gc.collect()  # Python garbage collection
    device_sync()  # Synchronize and clear device cache

# Global state for worker process - set once when process starts
_WORKER_MODULE_PATH = None
_WORKER_KERNEL_NAME = None
_WORKER_MODULE = None

def _bench_worker_init(module_path: str, kernel_name: str):
    """
    Initialize worker process by loading the kernel module.
    This runs once per worker process, avoiding repeated imports.
    """
    global _WORKER_MODULE_PATH, _WORKER_KERNEL_NAME, _WORKER_MODULE

    _WORKER_MODULE_PATH = module_path
    _WORKER_KERNEL_NAME = kernel_name

    # Install Inductor shims in worker process
    _install_triton_grid_shim()
    install_inductor_shims()

    # Import the module containing the kernel
    mod_name = f"worker_mod_{Path(module_path).stem}_{os.getpid()}"
    _WORKER_MODULE = import_from_path(mod_name, module_path)

def _bench_worker_main(module_path, kernel_name, grid, args, kernel_config, warmup, iters, repeat_kernel, result_queue):
    """
    Main entry point for worker process - initializes then runs benchmark.
    This is a module-level function so it can be pickled.
    """
    _bench_worker_init(module_path, kernel_name)
    _bench_worker(grid, args, kernel_config, warmup, iters, repeat_kernel, result_queue)

def _bench_worker(grid, args, kernel_config, warmup, iters, repeat_kernel, result_queue):
    """
    Worker function that runs in a separate process with its own GPU context.
    Loads the kernel from the pre-initialized module to avoid pickling issues.
    Uses triton.testing.do_bench for consistent benchmarking.
    """
    try:
        # Get kernel function from the module loaded during init
        if _WORKER_MODULE is None or _WORKER_KERNEL_NAME is None:
            raise RuntimeError("Worker process not initialized properly")

        kernel_fn = getattr(_WORKER_MODULE, _WORKER_KERNEL_NAME)

        def kernel_call():
            kernel_fn[grid](*args, **kernel_config)

        # Use triton.testing.do_bench for standardized benchmarking
        # warmup and rep (repetitions) parameters control the benchmark
        ms = triton.testing.do_bench(kernel_call, warmup=warmup, rep=iters)

        result_queue.put(ms)

    except Exception as e:
        result_queue.put(e)

def bench(kernel_fn, grid, args, kernel_config, warmup=25, iters=100, repeat_kernel=1, timeout=30,
          module_path=None, kernel_name=None):
    """
    Multiprocessing-based benchmarking with timeout protection.
    Uses spawn to create isolated GPU contexts that can be safely killed.
    Uses triton.testing.do_bench for standardized benchmarking.

    Args:
        kernel_fn: Triton kernel function (used for validation in parent process)
        grid: Grid dimensions for kernel launch
        args: Arguments to pass to the kernel
        kernel_config: Configuration dict with XBLOCK, YBLOCK, etc.
        warmup: Number of warmup iterations (default: 25, matching triton.testing.do_bench)
        iters: Number of timing iterations/repetitions (default: 100, matching triton.testing.do_bench)
        repeat_kernel: Unused (kept for API compatibility)
        timeout: Maximum time in seconds for the entire benchmark
        module_path: Path to the module file containing the kernel (for worker process)
        kernel_name: Name of the kernel function (for worker process)
    """

    if module_path is None or kernel_name is None:
        raise ValueError("module_path and kernel_name are required for multiprocessing bench")

    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()

    # Create process that will initialize itself and run the benchmark
    # _bench_worker_main loads the module and calls _bench_worker
    # Note: repeat_kernel is passed for API compatibility but not used (do_bench handles repetitions)
    p = ctx.Process(
        target=_bench_worker_main,
        args=(module_path, kernel_name, grid, args, kernel_config, warmup, iters, repeat_kernel, result_queue)
    )
    p.start()

    try:
        # Wait for result with timeout
        result = result_queue.get(timeout=timeout)
        p.join(timeout=5)  # Give process time to clean up

        if isinstance(result, Exception):
            log.debug(f"    Benchmark FAILED: {result}")
            return float('inf')

        log.debug(f"    Benchmark result: {result:.6f} ms")
        return result

    except queue.Empty:
        log.debug(f"    WARNING: Benchmark timed out after {timeout}s")

        # Kill the process forcefully
        if p.is_alive():
            p.kill()
            p.join(timeout=5)

        # Force cleanup in parent process
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
                if hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()
            gc.collect()
        except Exception as cleanup_error:
            log.debug(f"    Cleanup after timeout failed: {cleanup_error}")

        return float('inf')

    finally:
        # Ensure process is terminated
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
                p.join()

def _bench_legacy(fn, warmup=5, iters=10, repeat_kernel=1, timeout_per_iter=30):
    """
    Legacy benchmarking function for backward compatibility.
    Used when fn doesn't return kernel execution data tuple.
    """
    times = []

    # Extended warmup
    log.debug(f"    Warming up with {warmup} iterations...")
    for i in range(warmup):
        try:
            def warmup_fn():
                for _ in range(repeat_kernel):
                    fn()
                device_sync()
                return True

            with_timeout(warmup_fn, timeout_per_iter)
        except TimeoutException:
            log.debug(f"    WARNING: Warmup iteration {i+1} timed out, skipping this configuration")
            raise
        except Exception as e:
            log.debug(f"    WARNING: Warmup iteration {i+1} failed: {e}")
            raise

    log.debug(f"    Collecting {iters} timing samples...")
    # Collect timing samples
    for i in range(iters):
        try:
            if torch.cuda.is_available():
                def cuda_timing():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    torch.cuda.synchronize()  # Ensure clean start
                    start.record()

                    # Run kernel multiple times to increase measurement duration
                    for _ in range(repeat_kernel):
                        fn()

                    end.record()
                    torch.cuda.synchronize()  # Wait for GPU to finish

                    # Divide by repeat_kernel to get per-kernel time
                    total_time = start.elapsed_time(end)
                    per_kernel_time = total_time / repeat_kernel
                    return per_kernel_time

                timing_result = with_timeout(cuda_timing, timeout_per_iter)
                times.append(timing_result)
            else:
                def cpu_timing():
                    device_sync()  # Ensure clean start
                    start_time = time.perf_counter()

                    for _ in range(repeat_kernel):
                        fn()

                    device_sync()
                    end_time = time.perf_counter()
                    total_time = (end_time - start_time) * 1000.0
                    per_kernel_time = total_time / repeat_kernel
                    return per_kernel_time

                timing_result = with_timeout(cpu_timing, timeout_per_iter)
                times.append(timing_result)

        except TimeoutException:
            log.debug(f"    WARNING: Timing iteration {i+1} timed out")
            raise
        except Exception as e:
            log.debug(f"    WARNING: Timing iteration {i+1} failed: {e}")
            raise

    if not times:
        raise RuntimeError("No successful timing iterations completed")

    # Remove outliers (top and bottom 10%) for more stable results
    times.sort()
    n = len(times)
    trim_count = max(1, n // 10)  # Remove 10% from each end
    trimmed_times = times[trim_count:-trim_count] if n > 2 else times

    # Return median of trimmed data
    median_time = trimmed_times[len(trimmed_times) // 2]

    # Print statistics for debugging
    min_time = min(times)
    max_time = max(times)
    std_dev = (sum((t - median_time)**2 for t in trimmed_times) / len(trimmed_times))**0.5 if len(trimmed_times) > 1 else 0.0

    log.debug(f"    Timing stats: min={min_time:.6f}ms, median={median_time:.6f}ms, max={max_time:.6f}ms, std={std_dev:.6f}ms")

    return median_time

# ---------------------------
# 5) Grid helpers
# ---------------------------
def ceil_div(a, b):
    return (a + b - 1) // b

def grid_1d(xnumel: int, XBLOCK: int):
    return (ceil_div(xnumel, XBLOCK),)

def grid_2d(xnumel: int, ynumel: int, XBLOCK: int, YBLOCK: int):
    return (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK))

def detect_kernel_type(param_names: List[str]) -> str:
    """
    Kernel type detection that distinguishes reduction types.

    Returns:
        - '1d': Standard 1D pointwise kernels
        - '2d': 2D kernels with both X and Y dimensions
        - 'reduction': Reduction kernels with tunable R0_BLOCK parameter
        - 'reduction_fixed': Reduction kernels with fixed R0_BLOCK (not tunable)
        - 'unknown': Cannot determine type
    """
    has_ynumel = "ynumel" in param_names
    has_xnumel = "xnumel" in param_names
    has_r0_numel = "r0_numel" in param_names
    has_yblock = any("YBLOCK" in p for p in param_names)
    has_xblock = any("XBLOCK" in p for p in param_names)
    has_r0_block = any("R0_BLOCK" in p for p in param_names)

    # Check for reduction patterns - distinguish between tunable and fixed R0_BLOCK
    if has_r0_numel or has_r0_block:
        if has_r0_block:
            return "reduction"  # Has tunable R0_BLOCK parameter
        else:
            return "reduction_fixed"  # Has r0_numel but R0_BLOCK is fixed/predefined

    # Standard 2D kernels
    if has_ynumel and has_xnumel and has_yblock and has_xblock:
        return "2d"

    # Standard 1D kernels
    if has_xnumel and has_xblock:
        return "1d"

    # Unknown type
    return "unknown"

def _is_problematic_config(kernel_config: Dict, xnumel: int, kernel_type: str) -> bool:
    """
    Identify configurations likely to hang or fail based on empirical observations.
    """
    # Very large R0_BLOCK with small xnumel often causes hangs
    if kernel_type == "reduction":
        r0_block = kernel_config.get("R0_BLOCK", 0)
        xblock = kernel_config.get("XBLOCK", 0)
        if r0_block > 8192 and xnumel < 1024:
            return True
        # Very high warp count with large block sizes
        if kernel_config.get("num_warps", 0) > 16 and r0_block > 4096:
            return True

    # Extremely large block sizes for small problems
    if kernel_type in ["1d", "reduction_fixed"]:
        xblock = kernel_config.get("XBLOCK", 0)
        if xblock > xnumel:  # Block larger than problem size
            return True
        if xblock > 2048 and xnumel < 4096:  # Oversized blocks
            return True

    return False

def validate_kernel_args(kernel_fn, args, grid, kernel_config, kernel_name):
    """
    Validate that kernel arguments are compatible before benchmarking.
    Uses multiple validation strategies to catch problematic configs early.
    """
    try:
        # Strategy 1: Quick validation with minimal timeout
        def quick_test():
            kernel_fn[grid](*args, **kernel_config)
            device_sync()
            return True

        # Use very short timeout for validation (5 seconds)
        try:
            with_timeout(quick_test, 10)
        except TimeoutException:
            log.debug(f"    Validation timeout for {kernel_name} with config {kernel_config}")
            return False

        # Strategy 2: Check GPU memory state after execution
        if torch.cuda.is_available():
            try:
                # Check if we can still allocate a small tensor (GPU not hung)
                test_tensor = torch.zeros(10, device='cuda')
                del test_tensor
                torch.cuda.synchronize()
            except Exception as e:
                log.debug(f"    GPU state validation failed for {kernel_name}: {e}")
                return False

        return True

    except Exception as e:
        log.debug(f"    Validation failed for {kernel_name} with config {kernel_config}: {e}")
        return False

# ---------------------------
# 6) Tuning one kernel (comprehensive version)
# ---------------------------

import triton
import triton.language as tl

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
    time_output = times_ms[len(times_ms)//2]
    return time_output

def _parse_best_from_log(text: str):
    """
    Look for Triton's 'best config selected:' line. Returns dict or None.
    Example line:
      '... best config selected: XBLOCK: 32, num_warps: 1, num_ctas: 1, num_stages: 2, ...'
    """
    _BEST_LINE_RE = re.compile(
        r"best config selected:\s*XBLOCK:\s*(\d+),\s*num_warps:\s*(\d+).*?num_stages:\s*(\d+)",
        flags=re.IGNORECASE
    )
    m = _BEST_LINE_RE.search(text)
    if not m:
        return None
    xb, w, s = map(int, m.groups())
    return {"XBLOCK": xb, "num_warps": w, "num_stages": s}

def extract_default_config(src: str, kernel_name: str) -> Optional[Dict]:
    """
    Extract default configuration from kernel source code by:
    1. Locating the kernel definition
    2. Identifying tunable parameters (those marked with tl.constexpr)
    3. Finding kernel invocations and extracting actual values used

    Returns dict with BLOCK parameters and num_warps/num_stages or None if extraction fails
    """
    # Step 1: Find kernel definition and identify constexpr parameters
    kernel_pattern = rf"@triton\.jit\s+def\s+{re.escape(kernel_name)}\s*\(([^)]+)\):"
    kernel_match = re.search(kernel_pattern, src, re.DOTALL)

    if not kernel_match:
        log.warning(f"Could not find kernel definition for {kernel_name}")
        return None

    params_str = kernel_match.group(1)

    # Step 2: Parse parameters and identify which ones are constexpr (tunable)
    # Format: "param_name: tl.constexpr" or "PARAM: tl.constexpr"
    constexpr_params = []

    # Split parameters by comma, handling multi-line and cleaning whitespace
    # Replace multiple whitespace with single space, preserving structure
    cleaned_params = re.sub(r'\s+', '', params_str.strip())
    param_lines = cleaned_params.split(',')

    for param in param_lines:
        param = param.strip()
        if 'tl.constexpr' in param:
            # Extract parameter name (before the colon)
            param_name_match = re.search(r'(\w+)\s*:', param)
            if param_name_match:
                param_name = param_name_match.group(1)
                constexpr_params.append(param_name)

    if not constexpr_params:
        log.warning(f"No constexpr parameters found for kernel {kernel_name}")
        return None

    log.debug(f"Found constexpr parameters for {kernel_name}: {constexpr_params}")

    # Step 3: Find kernel invocation in call() function
    # Pattern: kernel_name[grid](args..., positional_values, num_warps=X, num_stages=Y)
    invocation_pattern = rf"{re.escape(kernel_name)}\[[^\]]+\]\(([^)]+)\)"
    invocation_matches = list(re.finditer(invocation_pattern, src))

    if not invocation_matches:
        log.warning(f"No invocations found for kernel {kernel_name}")
        return None

    # Use first invocation
    invocation = invocation_matches[0]
    args_str = invocation.group(1)

    # Step 4: Extract num_warps and num_stages from keyword arguments
    num_warps_match = re.search(r'num_warps\s*=\s*(\d+)', args_str)
    num_stages_match = re.search(r'num_stages\s*=\s*(\d+)', args_str)

    num_warps = int(num_warps_match.group(1)) if num_warps_match else None
    num_stages = int(num_stages_match.group(1)) if num_stages_match else None

    # Step 5: Extract positional argument values for constexpr parameters
    # Remove keyword arguments section to isolate positional args
    args_before_kwargs = re.split(r',\s*num_warps\s*=', args_str)[0]

    # Extract all numeric values from positional arguments
    # Split by comma and extract integers, filtering out variable names
    positional_values = []
    for arg in args_before_kwargs.split(','):
        arg = arg.strip()
        # Look for standalone numbers (not part of expressions or method calls)
        if re.match(r'^\d+$', arg):
            positional_values.append(int(arg))

    log.debug(f"Positional numeric values: {positional_values}")

    # Step 6: Map constexpr parameters to their values
    # The last N positional numeric values correspond to the constexpr parameters
    # (in the order they appear in the function signature)
    default_config = {}

    if len(positional_values) >= len(constexpr_params):
        # Take the last N values where N = number of constexpr params
        constexpr_values = positional_values[-len(constexpr_params):]

        for param_name, value in zip(constexpr_params, constexpr_values):
            default_config[param_name] = value
            log.debug(f"Mapped {param_name} = {value}")
    else:
        log.warning(f"Not enough positional values ({len(positional_values)}) for constexpr params ({len(constexpr_params)})")
        return None

    # Add num_warps and num_stages
    if num_warps is not None:
        default_config['num_warps'] = num_warps
    if num_stages is not None:
        default_config['num_stages'] = num_stages

    if default_config:
        log.debug(f"Successfully extracted default config for {kernel_name}: {default_config}")
        return default_config

    log.warning(f"Failed to extract default config for kernel {kernel_name}")
    return None

def tune_kernel(kernel_fn,
                kernel_name: str,
                numel_param_names: List[str],
                numel_values: List[int],
                other_param_names: List[str],
                device: torch.device,
                threshold_pct: float = 2.0,           # within X% of best
                top_k: Optional[int] = None,          # cap size of near-opt set (optional)
                save_json: Optional[str] = None,      # path to save results (optional)
                timeout_per_config: int = 60,         # max time per config in seconds
                source_code: Optional[str] = None,    # source code for extracting default config
                module_path: Optional[str] = None     # path to module file (for multiprocessing)
               ) -> Dict:
    """
    Exhaustively time each (XBLOCK, num_warps, num_stages) config, then
    return the best and all configs whose latency <= best * (1 + threshold_pct/100).

    Args:
        numel_param_names: List of parameter names ending with 'numel'
        numel_values: List of corresponding values for numel parameters
        other_param_names: List of non-numel parameter names
        module_path: Path to module file containing the kernel (required for multiprocessing)

    Notes:
      - Uses your existing `bench`, `grid_1d`, and `make_dummy_args_for_kernel`.
      - Warmup inside `bench` ensures JIT compile doesn't pollute timings.
      - Includes timeout mechanism to skip slow/hanging kernels.
    """
    # ---- Inputs ----
    args, _ = make_dummy_args_for_kernel(numel_param_names, numel_values, other_param_names, device)

    # Combine all parameter names for compatibility
    param_names = numel_param_names + other_param_names

    # Get primary numel for grid calculations (prefer xnumel, fallback to first/largest)
    numel_dict = dict(zip(numel_param_names, numel_values))
    xnumel = numel_dict.get("xnumel", max(numel_values) if numel_values else 16384)

    # Detect kernel type
    kernel_type = detect_kernel_type(param_names)
    log.debug(f"  Detected kernel type: {kernel_type}")

    # Search space - adjust based on kernel type
    YBLOCKS = []  # Initialize to avoid unbound variable
    R0_BLOCKS = []  # Initialize to avoid unbound variable
    ynumel = None  # Will be set for 2D kernels

    if kernel_type == "2d":
        # Based on analysis: XBLOCK=32, YBLOCK=32 (64%), XBLOCK=16, YBLOCK=64 (18.6%)
        XBLOCKS = [16, 32, 64, 128]  # Focus around common values
        YBLOCKS = [4, 32, 64]        # Include actual usage patterns
        WARPS   = [2, 4, 8]          # num_warps=4 dominates (95.4%)
        STAGES  = [1, 2]             # num_stages=1 is universal

        # For 2D kernels, ynumel is required
        if "ynumel" in numel_param_names:
            ynumel_idx = numel_param_names.index("ynumel")
            ynumel = numel_values[ynumel_idx]
        else:
            raise RuntimeError(f"2D kernel {kernel_name} requires 'ynumel' parameter but it was not found in numel_param_names: {numel_param_names}")

        # Type assertion since we know ynumel is set for 2D kernels
        assert ynumel is not None
    elif kernel_type == "reduction":
        # Based on analysis: R0_BLOCK=2048, XBLOCK=1 (96.9%), num_warps=16 dominates
        XBLOCKS = [1, 64, 128]       # XBLOCK=1 is dominant, explore alternatives
        R0_BLOCKS = [1024, 2048, 4096, 8192]  # Focus on larger R0_BLOCK values
        WARPS   = [4, 8, 16, 32]        # num_warps=16 dominates, explore nearby
        STAGES  = [1, 2]             # num_stages=1 is universal
    elif kernel_type == "1d":
        # Based on analysis: XBLOCK=1024 (34.4%), 512 (31.2%), 256 (19.1%)
        XBLOCKS = [256, 512, 1024, 2048]  # Focus on actual usage + explore larger
        WARPS   = [4, 8, 16]              # num_warps=4 dominates, explore higher
        STAGES  = [1, 2]                  # num_stages=1 is universal
    elif kernel_type == "reduction_fixed":
        # Based on analysis: XBLOCK=32 (41.7%), 8 (29.6%), 1 (24.9%)
        XBLOCKS = [1, 8, 16, 32, 64]     # Focus on small XBLOCK values
        WARPS   = [2, 4, 8]              # num_warps=8 dominates, explore nearby
        STAGES  = [1, 2]                 # num_stages=1 is universal
    else:
        # Unknown kernel type - fallback to conservative tuning
        XBLOCKS = [64, 128, 256, 512]
        WARPS   = [2, 4, 8]
        STAGES  = [1, 2]

    # Extract default config if source code is provided
    default_config = None
    default_config_extraction_failed = False
    if source_code:
        default_config = extract_default_config(source_code, kernel_name)
        if default_config is None:
            default_config_extraction_failed = True
            log.warning(f"Failed to extract default config for {kernel_name}")
        else:
            log.debug(f"Extracted default config for {kernel_name}: {default_config}")

    results: List[Dict] = []
    all_configs: List[Dict] = []  # Store all attempted configs
    attempts = 0
    errors = 0
    timeouts = 0

    # Configure timeouts - use triton.testing.do_bench defaults
    warmup_iters = 5   # Default for triton.testing.do_bench
    timing_iters = 10  # Default for triton.testing.do_bench (rep parameter)
    log.debug(f"  Using timeout: {timeout_per_config}s per config (warmup={warmup_iters}, rep={timing_iters})")

    # ---- Record search space ----
    search_space = {
        "kernel_type": kernel_type,
        "param_names": param_names,
        "numel_dict": numel_dict
    }

    if kernel_type == "2d":
        search_space["XBLOCKS"] = XBLOCKS
        search_space["YBLOCKS"] = YBLOCKS
        search_space["WARPS"] = WARPS
        search_space["STAGES"] = STAGES
    elif kernel_type == "reduction":
        search_space["XBLOCKS"] = XBLOCKS
        search_space["R0_BLOCKS"] = R0_BLOCKS
        search_space["WARPS"] = WARPS
        search_space["STAGES"] = STAGES
    elif kernel_type in ["1d", "reduction_fixed"]:
        search_space["XBLOCKS"] = XBLOCKS
        search_space["WARPS"] = WARPS
        search_space["STAGES"] = STAGES

    # ---- Sweep all configs ----
    if kernel_type == "2d":
        # ynumel is guaranteed to be set for 2D kernels due to the error check above
        ynumel_val: int = ynumel  # type: ignore
        for yb in YBLOCKS:
            for xb in XBLOCKS:
                g = grid_2d(xnumel, ynumel_val, xb, yb)
                for w in WARPS:
                    for s in STAGES:
                        attempts += 1
                        kernel_config = {"YBLOCK": yb, "XBLOCK": xb, "num_warps": w, "num_stages": s}

                        # Record this config attempt
                        config_record = {
                            "config": kernel_config.copy(),
                            "status": "pending",
                            "lat_ms": None,
                            "error": None
                        }

                        try:

                            # Validate kernel call before benchmarking
                            if not validate_kernel_args(kernel_fn, args, g, kernel_config, kernel_name):
                                raise RuntimeError("Kernel validation failed")

                            repeat_count = max(1, int(10.0 / max(0.001, xnumel * ynumel_val / 100000)))
                            lat_ms = bench(kernel_fn, g, args, kernel_config, warmup=warmup_iters, iters=timing_iters,
                                         repeat_kernel=repeat_count, timeout=timeout_per_config,
                                         module_path=module_path, kernel_name=kernel_name)

                            results.append({
                                "lat_ms": float(lat_ms),
                                "cfg": {"YBLOCK": yb, "XBLOCK": xb, "num_warps": w, "num_stages": s, "grid": g},
                            })
                            config_record["status"] = "success"
                            config_record["lat_ms"] = float(lat_ms)
                            log.debug(f"  tried: {{'YBLOCK': {yb}, 'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}, 'repeat': {repeat_count}}} -> {lat_ms:.6f} ms")
                        except TimeoutException:
                            config_record["status"] = "timeout"
                            config_record["error"] = "Execution timed out"
                            log.debug(f"  TIMEOUT: {{'YBLOCK': {yb}, 'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}}} -> skipping")
                            timeouts += 1
                        except Exception as e:
                            config_record["status"] = "error"
                            config_record["error"] = str(e)
                            log.debug(f"  ERROR: {{'YBLOCK': {yb}, 'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}}} -> {e}")
                            errors += 1

                        all_configs.append(config_record)

                        # Cleanup memory after each config to prevent OOM
                        cleanup_memory()

    elif kernel_type == "reduction" or any("R0_BLOCK" in p for p in param_names):
        # Reduction kernels with R0_BLOCK parameter
        for xb in XBLOCKS:
            g = grid_1d(xnumel, xb)
            for r0b in R0_BLOCKS:
                for w in WARPS:
                    for s in STAGES:
                        attempts += 1
                        kernel_config = {"XBLOCK": xb, "R0_BLOCK": r0b, "num_warps": w, "num_stages": s}

                        # Record this config attempt
                        config_record = {
                            "config": kernel_config.copy(),
                            "status": "pending",
                            "lat_ms": None,
                            "error": None
                        }

                        # Skip potentially problematic configurations
                        # if _is_problematic_config(kernel_config, xnumel, kernel_type):
                        #     config_record["status"] = "skipped"
                        #     config_record["error"] = "Skipped potentially problematic configuration"
                        #     log.debug(f"  SKIP: {kernel_config} -> likely to hang or fail")
                        #     all_configs.append(config_record)
                        #     cleanup_memory()
                        #     continue

                        try:
                            # Validate kernel call before benchmarking
                            if not validate_kernel_args(kernel_fn, args, g, kernel_config, kernel_name):
                                raise RuntimeError("Reduction kernel validation failed")

                            repeat_count = max(1, int(10.0 / max(0.001, xnumel / 100000)))
                            lat_ms = bench(kernel_fn, g, args, kernel_config, warmup=warmup_iters, iters=timing_iters,
                                         repeat_kernel=repeat_count, timeout=timeout_per_config,
                                         module_path=module_path, kernel_name=kernel_name)
                            results.append({
                                "lat_ms": float(lat_ms),
                                "cfg": {"XBLOCK": xb, "R0_BLOCK": r0b, "num_warps": w, "num_stages": s, "grid": g},
                            })
                            config_record["status"] = "success"
                            config_record["lat_ms"] = float(lat_ms)
                            log.debug(f"  tried: {{'XBLOCK': {xb}, 'R0_BLOCK': {r0b}, 'num_warps': {w}, 'num_stages': {s}, 'repeat': {repeat_count}}} -> {lat_ms:.6f} ms")
                        except TimeoutException:
                            config_record["status"] = "timeout"
                            config_record["error"] = "Execution timed out"
                            log.debug(f"  TIMEOUT: {{'XBLOCK': {xb}, 'R0_BLOCK': {r0b}, 'num_warps': {w}, 'num_stages': {s}}} -> skipping")
                            timeouts += 1
                        except Exception as e:
                            config_record["status"] = "error"
                            config_record["error"] = str(e)
                            log.debug(f"  ERROR: {{'XBLOCK': {xb}, 'R0_BLOCK': {r0b}, 'num_warps': {w}, 'num_stages': {s}}} -> {e}")
                            errors += 1

                        all_configs.append(config_record)

                        # Cleanup memory after each config to prevent OOM
                        cleanup_memory()

    elif kernel_type == "1d":
        # Regular 1D kernels
        for xb in XBLOCKS:
            g = grid_1d(xnumel, xb)
            for w in WARPS:
                for s in STAGES:
                    attempts += 1
                    kernel_config = {"XBLOCK": xb, "num_warps": w, "num_stages": s}

                    # Record this config attempt
                    config_record = {
                        "config": kernel_config.copy(),
                        "status": "pending",
                        "lat_ms": None,
                        "error": None
                    }

                    # Skip potentially problematic configurations
                    # if _is_problematic_config(kernel_config, xnumel, kernel_type):
                    #     config_record["status"] = "skipped"
                    #     config_record["error"] = "Skipped potentially problematic configuration"
                    #     log.debug(f"  SKIP: {kernel_config} -> likely to hang or fail")
                    #     all_configs.append(config_record)
                    #     cleanup_memory()
                    #     continue

                    try:
                        # Validate kernel call before benchmarking
                        if not validate_kernel_args(kernel_fn, args, g, kernel_config, kernel_name):
                            raise RuntimeError("1D kernel validation failed")

                        repeat_count = max(1, int(10.0 / max(0.001, xnumel / 100000)))
                        lat_ms = bench(kernel_fn, g, args, kernel_config, warmup=warmup_iters, iters=timing_iters,
                                         repeat_kernel=repeat_count, timeout=timeout_per_config,
                                         module_path=module_path, kernel_name=kernel_name)

                        results.append({
                            "lat_ms": float(lat_ms),
                            "cfg": {"XBLOCK": xb, "num_warps": w, "num_stages": s, "grid": g},
                        })
                        config_record["status"] = "success"
                        config_record["lat_ms"] = float(lat_ms)
                        log.debug(f"  tried: {{'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}, 'repeat': {repeat_count}}} -> {lat_ms:.6f} ms")
                    except TimeoutException:
                        config_record["status"] = "timeout"
                        config_record["error"] = "Execution timed out"
                        log.debug(f"  TIMEOUT: {{'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}}} -> skipping")
                        timeouts += 1
                    except Exception as e:
                        config_record["status"] = "error"
                        config_record["error"] = str(e)
                        log.debug(f"  ERROR: {{'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}}} -> {e}")
                        errors += 1

                    all_configs.append(config_record)

                    # Cleanup memory after each config to prevent OOM
                    cleanup_memory()

    elif kernel_type == "reduction_fixed":
        # Reduction_fixed kernels (like triton_per_* with small XBLOCK values)
        for xb in XBLOCKS:
            g = grid_1d(xnumel, xb)
            for w in WARPS:
                for s in STAGES:
                    attempts += 1
                    kernel_config = {"XBLOCK": xb, "num_warps": w, "num_stages": s}

                    # Record this config attempt
                    config_record = {
                        "config": kernel_config.copy(),
                        "status": "pending",
                        "lat_ms": None,
                        "error": None
                    }

                    try:
                        # Validate kernel call before benchmarking
                        if not validate_kernel_args(kernel_fn, args, g, kernel_config, kernel_name):
                            raise RuntimeError("Reduction_fixed kernel validation failed")

                        repeat_count = max(1, int(10.0 / max(0.001, xnumel / 100000)))
                        lat_ms = bench(kernel_fn, g, args, kernel_config, warmup=warmup_iters, iters=timing_iters,
                                         repeat_kernel=repeat_count, timeout=timeout_per_config,
                                         module_path=module_path, kernel_name=kernel_name)

                        results.append({
                            "lat_ms": float(lat_ms),
                            "cfg": {"XBLOCK": xb, "num_warps": w, "num_stages": s, "grid": g},
                        })
                        config_record["status"] = "success"
                        config_record["lat_ms"] = float(lat_ms)
                        log.debug(f"  tried: {{'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}, 'repeat': {repeat_count}}} -> {lat_ms:.6f} ms")
                    except TimeoutException:
                        config_record["status"] = "timeout"
                        config_record["error"] = "Execution timed out"
                        log.debug(f"  TIMEOUT: {{'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}}} -> skipping")
                        timeouts += 1
                    except Exception as e:
                        config_record["status"] = "error"
                        config_record["error"] = str(e)
                        log.debug(f"  ERROR: {{'XBLOCK': {xb}, 'num_warps': {w}, 'num_stages': {s}}} -> {e}")
                        errors += 1

                    all_configs.append(config_record)

                    # Cleanup memory after each config to prevent OOM
                    cleanup_memory()

    else:
        # Unknown kernel type - skip tuning and count as error
        log.warning(f"Unknown kernel type '{kernel_type}' for {kernel_name}, skipping tuning")
        errors += 1

    log.debug(f"  Summary: {attempts} attempts, {len(results)} successful, {errors} errors, {timeouts} timeouts")

    if not results:
        raise RuntimeError(f"[{kernel_name}] No valid configs (attempted {attempts}, errors {errors}, timeouts {timeouts}).")

    # ---- Rank & threshold ----
    results.sort(key=lambda r: r["lat_ms"])
    best = results[0]
    best_time = best["lat_ms"]
    thresh = best_time * (1.0 + (threshold_pct / 100.0))

    near_opt = [r for r in results if r["lat_ms"] <= thresh]
    if top_k is not None:
        near_opt = near_opt[:top_k]

    # Backward compatibility: flatten selected meta-params from cfg into best dict
    if isinstance(best, dict) and "cfg" in best:
        flat_best = dict(best)  # shallow copy
        cfg = best.get("cfg", {})
        for k in ["XBLOCK", "YBLOCK", "R0_BLOCK", "num_warps", "num_stages"]:
            if k in cfg and k not in flat_best:
                flat_best[k] = cfg[k]
    else:
        flat_best = best

    out = {
        "kernel": kernel_name,
        "xnumel": xnumel,
        "ynumel": ynumel if kernel_type == "2d" else None,
        "grid": flat_best.get("cfg", {}).get("grid") if "cfg" in flat_best else None,
        "attempts": attempts,
        "errors": errors,
        "timeouts": timeouts,
        "successful": len(results),
        "best": flat_best,
        "near_opt_threshold_pct": threshold_pct,
        "near_opt_count": len(near_opt),
        "near_opt": near_opt,
        "default_config": default_config,
        "default_config_extraction_failed": default_config_extraction_failed,
        "search_space": search_space,
        "all_configs": all_configs,
    }

    if save_json:
        with open(save_json, "w") as f:
            json.dump(out, f, indent=2)
        log.debug(f"  Saved tuning results to {save_json}")

    return out

# ---------------------------
# 7) Walk a module and tune
# ---------------------------
def autotune_file(py_path: Path, device: Optional[str] = None, timeout_per_config: int = 60, logger: Optional[logging.Logger] = None):
    """
    Auto-tune all kernels in a Triton module file.

    Args:
        py_path: Path to the Triton kernel file
        device: Target device ('cuda', 'cpu', etc.)
        timeout_per_config: Maximum time in seconds to spend on each config within a kernel (default: 60 seconds)
        logger: Logger instance to use (if None, creates a default logger)
    """
    # Use provided logger or fall back to module logger
    if logger is None:
        logger = logging.getLogger(__name__)
    # Create (or reuse) a performance directory in current working directory
    perf_dir = Path.cwd() / "performance"
    perf_dir.mkdir(exist_ok=True)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mod_name = f"tune_mod_{Path(py_path).stem}"
    logger.info(f"Processing file: {py_path}")
    get_inputs(str(py_path))
    # exit()
    src = read_text(Path(py_path))
    m = import_from_path(mod_name, str(py_path))

    # for name, obj in vars(m).items():
    #     log.debug(f"{name}: {type(obj)}")
    # exit()

    results = []
    total_kernels = 0
    successful_kernels = 0

    for name, obj in vars(m).items():
        # Inductor kernels are jit-compiled callables (decorated with @triton.jit)
        if callable(obj) and name.startswith("triton_"):
            total_kernels += 1
            numel_param_names, numel_values, other_param_names = infer_numel_for_kernel(src, name)

            if not numel_param_names or not numel_values or not other_param_names:
                logger.debug(f"[SKIP] Could not infer parameters for {name} (numel_params={numel_param_names}, numel_values={numel_values}, other_params={other_param_names})")
                continue

            logger.info(f"[TUNE] {name}  (numel_params={dict(zip(numel_param_names, numel_values))}) - Max time: {timeout_per_config}s")

            try:
                # Call tune_kernel with the new parameter structure, passing source code and module path
                res = tune_kernel(obj, name, numel_param_names, numel_values, other_param_names, device_obj,
                                save_json=str(perf_dir / f"{name}_tune.json"),
                                timeout_per_config=timeout_per_config,  # 1 minute per config within the kernel
                                source_code=src,  # Pass source code for default config extraction
                                module_path=str(py_path))  # Pass module path for multiprocessing
                logger.info(f"   best: {res['best']}")
                if res.get('default_config'):
                    logger.info(f"   default: {res['default_config']}")
                logger.info(f"   total configs tried: {len(res['all_configs'])}")
                results.append(res)
                successful_kernels += 1

            except Exception as e:
                logger.error(f"[ERROR] Failed to tune {name}: {e}")
                # Continue with next kernel instead of failing completely
                continue

    logger.info("=== SUMMARY ===")
    logger.info(f"Total kernels found: {total_kernels}")
    logger.info(f"Successfully tuned: {successful_kernels}")
    logger.info(f"Failed/Skipped: {total_kernels - successful_kernels}")

    return results
# Configure logging only if running as main script
# When imported by auto_tune_all.py, use the existing logging configuration
if __name__ == "__main__":
    # Running as standalone script - set up our own logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    # When imported, check if logging is already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # No handlers configured yet, set up minimal logging
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    # If handlers exist, the parent module (like auto_tune_all.py) has already configured logging

log = logging.getLogger(__name__)
if __name__ == "__main__":
    # Running as standalone script - set up our own logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # Example usage
    # python tune_one_file.py /path/to/that_triton_file.py
    path = sys.argv[1]
    install_inductor_shims()
    out = autotune_file(Path(path))
    log.debug(out)
else:
    # When imported, check if logging is already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # No handlers configured yet, set up minimal logging
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    # If handlers exist, the parent module (like auto_tune_all.py) has already configured logging
