"""
Cross-process GPU benchmark lock.

Ensures only one benchmark runs on the GPU at a time, even when multiple
kernel-optimization subprocesses run in parallel.  Uses filelock (already
a project dependency via create_dataset.py).

The lock file path is resolved from:
  1. GPU_BENCHMARK_LOCK_PATH environment variable (if set)
  2. /tmp/triton_gpu_benchmark.lock (default)

batch_runner.py sets GPU_BENCHMARK_LOCK_PATH before spawning subprocesses,
so all children automatically inherit the same lock path.
"""

import logging
import os
from filelock import FileLock

log = logging.getLogger(__name__)

_DEFAULT_LOCK_PATH = "/tmp/triton_gpu_benchmark.lock"
_lock_instance = None


def get_gpu_lock() -> FileLock:
    """Return a singleton FileLock for GPU benchmarking.

    The lock is reusable and safe to acquire/release multiple times within
    the same process.  On Unix, ``filelock`` uses ``fcntl.flock()`` which
    is automatically released when the process exits (even via SIGKILL),
    so stale lock files are never a problem.
    """
    global _lock_instance
    if _lock_instance is None:
        lock_path = os.environ.get("GPU_BENCHMARK_LOCK_PATH", _DEFAULT_LOCK_PATH)
        _lock_instance = FileLock(lock_path, timeout=-1)
        log.debug("GPU benchmark lock: %s", lock_path)
    return _lock_instance
