"""
Benchmarking utilities for Triton kernel execution.

Provides robust benchmarking infrastructure using multiprocessing isolation
and triton.testing.do_bench for accurate timing measurements.
"""

import gc
import logging
import multiprocessing
import os
import queue
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Callable, Optional, List

import torch
import triton
import triton.testing

from gpu_lock import get_gpu_lock

log = logging.getLogger(__name__)


def has_xpu():
	"""Check if Intel XPU is available"""
	return hasattr(torch, "xpu") and torch.xpu.is_available()


def device_sync(device: str = "cuda"):
	"""Synchronize device and clear cache"""
	if device == "cuda" and torch.cuda.is_available():
		torch.cuda.synchronize()
		if hasattr(torch.cuda, 'empty_cache'):
			torch.cuda.empty_cache()
	elif device == "xpu" and has_xpu():
		torch.xpu.synchronize()
		if hasattr(torch.xpu, 'empty_cache'):
			torch.xpu.empty_cache()


def cleanup_memory(device: str = "cuda"):
	"""Aggressive memory cleanup to prevent OOM"""
	gc.collect()
	device_sync(device)


# Global state for worker process
_WORKER_MODULE = None
_WORKER_KERNEL_NAME = None


def _bench_worker_init(module_path: str, kernel_name: str):
	"""
	Initialize worker process by loading the kernel module.
	Runs once per worker process.
	"""
	global _WORKER_MODULE, _WORKER_KERNEL_NAME

	import importlib.util
	import sys

	_WORKER_KERNEL_NAME = kernel_name

	# Import the module containing the kernel
	spec = importlib.util.spec_from_file_location(
		f"kernel_module_{os.getpid()}",
		module_path
	)
	_WORKER_MODULE = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = _WORKER_MODULE
	spec.loader.exec_module(_WORKER_MODULE)


def _bench_worker_main(module_path: str, kernel_name: str, grid: Tuple[int, ...],
					   args: Tuple, kwargs: Dict, warmup: int, rep: int,
					   result_queue: multiprocessing.Queue):
	"""
	Main entry point for worker process.
	This is a module-level function so it can be pickled.
	"""
	_bench_worker_init(module_path, kernel_name)
	_bench_worker(grid, args, kwargs, warmup, rep, result_queue)


def _bench_worker(grid: Tuple[int, ...], args: Tuple, kwargs: Dict,
				  warmup: int, rep: int, result_queue: multiprocessing.Queue):
	"""
	Worker function that runs in a separate process with isolated GPU context.
	Uses triton.testing.do_bench for standardized benchmarking.
	"""
	try:
		if _WORKER_MODULE is None or _WORKER_KERNEL_NAME is None:
			raise RuntimeError("Worker process not initialized properly")

		kernel_fn = getattr(_WORKER_MODULE, _WORKER_KERNEL_NAME)

		# Create kernel call closure
		def kernel_call():
			kernel_fn[grid](*args, **kwargs)

		# Use triton.testing.do_bench for standardized benchmarking
		ms = triton.testing.do_bench(kernel_call, warmup=warmup, rep=rep, return_mode="all")

		result_queue.put(ms)

	except Exception as e:
		result_queue.put(e)


def benchmark_kernel(
	module_path: Path,
	kernel_name: str,
	grid: Tuple[int, ...],
	args: Tuple,
	kwargs: Dict[str, Any],
	warmup: int = 25,
	rep: int = 100,
	timeout: int = 30,
	device: str = "cuda"
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float]]:
	"""
	Benchmark a Triton kernel with multiprocessing isolation.

	Uses spawn to create isolated GPU contexts that can be safely killed if timeout occurs.
	Uses triton.testing.do_bench for standardized benchmarking.

	Args:
		module_path: Path to the Python file containing the kernel
		kernel_name: Name of the kernel function
		grid: Grid dimensions for kernel launch (e.g., (64,) or (32, 16))
		args: Positional arguments for the kernel (tensors, scalars)
		kwargs: Keyword arguments for the kernel (XBLOCK, YBLOCK, num_warps, etc.)
		warmup: Number of warmup iterations (default: 25)
		rep: Number of timing iterations/repetitions (default: 100)
		timeout: Maximum time in seconds for the entire benchmark (default: 30)
		device: Device to use ("cuda", "xpu", "cpu")

	Returns:
		Median execution time in milliseconds
		Returns float('inf') if benchmark fails or times out

	Example:
		>>> result = benchmark_kernel(
		...     module_path=Path("kernel.py"),
		...     kernel_name="triton_poi_fused_add_0",
		...     grid=(256,),
		...     args=(out_tensor, in_tensor, 16384),
		...     kwargs={'XBLOCK': 256, 'num_warps': 4, 'num_stages': 2},
		...     warmup=25,
		...     rep=100
		... )
		>>> print(f"Kernel took {result:.4f} ms")
	"""

	if not module_path.exists():
		log.error(f"Module path does not exist: {module_path}")
		return float('inf')

	# Convert Path to string for pickling
	module_path_str = str(module_path.absolute())

	# Acquire GPU lock so only one benchmark runs at a time when multiple
	# kernel-optimization subprocesses execute in parallel.
	gpu_lock = get_gpu_lock()
	with gpu_lock:
		ctx = multiprocessing.get_context('spawn')
		result_queue = ctx.Queue()

		# Create worker process
		p = ctx.Process(
			target=_bench_worker_main,
			args=(module_path_str, kernel_name, grid, args, kwargs, warmup, rep, result_queue)
		)
		p.start()

		try:
			# Wait for result with timeout
			result = result_queue.get(timeout=timeout)

			# Filter to use only 20%-80% of results (remove outliers)
			if isinstance(result, (list, tuple)) and len(result) > 4:
				sorted_result = sorted(result)
				start_idx = int(len(sorted_result) * 0.2)
				end_idx = int(len(sorted_result) * 0.8)
				result = sorted_result[start_idx:end_idx]

			p.join(timeout=10)  # Give process time to clean up

			if isinstance(result, Exception):
				log.debug(f"Benchmark FAILED for {kernel_name}: {result}")
				return float('inf'), None, None, None, None

			# result is a list of timing measurements from return_mode="all"
			if isinstance(result, (list, tuple)):
				# Compute statistics
				min_time = min(result)
				max_time = max(result)
				mean_time = sum(result) / len(result)
				sorted_result = sorted(result)
				median_time = sorted_result[len(sorted_result) // 2]
				variance = sum((t - mean_time) ** 2 for t in result) / len(result)
				std_dev = variance ** 0.5

				log.debug(f"Benchmark statistics for {kernel_name}:")
				log.debug(f"  Min: {min_time:.6f} ms")
				log.debug(f"  Max: {max_time:.6f} ms")
				log.debug(f"  Median: {median_time:.6f} ms")
				log.debug(f"  Mean: {mean_time:.6f} ms")
				log.debug(f"  Std Dev: {std_dev:.6f} ms")
				log.debug(f"  Samples: {len(result)}")

				return median_time, min_time, max_time, mean_time, std_dev
			else:
				# Single value returned (shouldn't happen with return_mode="all")
				log.debug(f"Benchmark result for {kernel_name}: {result:.6f} ms")
				return result, None, None, None, None

		except queue.Empty:
			log.warning(f"Benchmark timed out after {timeout}s for {kernel_name}")

			# Kill the process forcefully
			if p.is_alive():
				p.kill()
				p.join(timeout=5)

			# Force cleanup in parent process
			try:
				cleanup_memory(device)
			except Exception as cleanup_error:
				log.debug(f"Cleanup after timeout failed: {cleanup_error}")

			return float('inf'), None, None, None, None

		finally:
			# Ensure process is terminated
			if p.is_alive():
				p.terminate()
				p.join(timeout=2)
				if p.is_alive():
					p.kill()
					p.join()


def benchmark_kernel_simple(
	kernel_fn: Callable,
	grid: Tuple[int, ...],
	args: Tuple,
	kwargs: Dict[str, Any],
	warmup: int = 25,
	rep: int = 100,
	device: str = "cuda"
) -> float:
	"""
	Simple in-process benchmark for quick testing (no isolation).

	This is less robust than benchmark_kernel() but useful for quick tests
	where multiprocessing overhead is not desired.

	Args:
		kernel_fn: Triton JIT kernel function
		grid: Grid dimensions for kernel launch
		args: Positional arguments for the kernel
		kwargs: Keyword arguments for the kernel
		warmup: Number of warmup iterations (default: 25)
		rep: Number of timing iterations (default: 100)
		device: Device to use ("cuda", "xpu", "cpu")

	Returns:
		Median execution time in milliseconds
	"""

	def kernel_call():
		kernel_fn[grid](*args, **kwargs)

	try:
		ms = triton.testing.do_bench(kernel_call, warmup=warmup, rep=rep)
		return ms
	except Exception as e:
		log.error(f"Simple benchmark failed: {e}")
		return float('inf')


def benchmark_with_statistics(
	module_path: Path,
	kernel_name: str,
	grid: Tuple[int, ...],
	args: Tuple,
	kwargs: Dict[str, Any],
	num_iterations: int = 100,
	warmup: int = 3,
	timeout: int = 30,
	device: str = "cuda"
) -> Dict[str, Any]:
	"""
	Benchmark kernel and return detailed statistics.

	Unlike benchmark_kernel which uses triton.testing.do_bench internally,
	this function performs manual timing to collect detailed statistics.

	Args:
		module_path: Path to the Python file containing the kernel
		kernel_name: Name of the kernel function
		grid: Grid dimensions for kernel launch
		args: Positional arguments for the kernel
		kwargs: Keyword arguments for the kernel
		num_iterations: Number of timing iterations
		warmup: Number of warmup iterations
		timeout: Maximum time in seconds
		device: Device to use

	Returns:
		Dictionary with detailed statistics including mean, median, min, max, std_dev
	"""

	# Import kernel module
	import importlib.util
	spec = importlib.util.spec_from_file_location("kernel_module", str(module_path))
	kernel_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(kernel_module)

	kernel_fn = getattr(kernel_module, kernel_name)

	# Warm-up
	log.debug(f"Warming up with {warmup} iterations...")
	for _ in range(warmup):
		try:
			kernel_fn[grid](*args, **kwargs)
			device_sync(device)
		except Exception as e:
			log.warning(f"Warm-up iteration failed: {e}")
			return {
				"success": False,
				"error": str(e),
				"device": device
			}

	# Benchmark
	log.debug(f"Benchmarking with {num_iterations} iterations...")
	execution_times = []

	for i in range(num_iterations):
		device_sync(device)
		start_time = time.perf_counter()

		try:
			kernel_fn[grid](*args, **kwargs)
		except Exception as e:
			log.error(f"Iteration {i} failed: {e}")
			continue

		device_sync(device)
		end_time = time.perf_counter()
		execution_times.append((end_time - start_time) * 1000)  # Convert to ms

	if not execution_times:
		return {
			"success": False,
			"error": "No successful iterations",
			"device": device
		}

	# Calculate statistics
	mean_time = sum(execution_times) / len(execution_times)
	min_time = min(execution_times)
	max_time = max(execution_times)
	median_time = sorted(execution_times)[len(execution_times) // 2]
	variance = sum((t - mean_time) ** 2 for t in execution_times) / len(execution_times)
	std_dev = variance ** 0.5

	return {
		"success": True,
		"device": device,
		"num_iterations": len(execution_times),
		"execution_time_ms": {
			"mean": mean_time,
			"median": median_time,
			"min": min_time,
			"max": max_time,
			"std_dev": std_dev
		},
		"throughput_calls_per_sec": 1000.0 / mean_time if mean_time > 0 else 0,
		"all_times": execution_times
	}


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
	
	# Reduction kernels
	if has_r0_numel:
		if has_r0_block:
			return 'reduction'
		else:
			return 'reduction_fixed'
	
	# 2D kernels
	if has_ynumel and (has_yblock or has_xblock):
		return '2d'
	
	# 1D kernels
	if has_xnumel and has_xblock:
		return '1d'
	
	return 'unknown'
