"""
GPU Architecture Knowledge Base

Defines architecture specifications for different GPU models
to be used by optimization agents.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GPUArchitecture:
    """GPU architecture specifications"""

    name: str
    compute_capability: str

    # SM (Streaming Multiprocessor) specs
    num_sms: int
    max_warps_per_sm: int
    max_threads_per_sm: int
    max_threads_per_block: int
    warp_size: int = 32

    # Memory specs
    shared_memory_per_sm_kb: int
    l1_cache_size_kb: int
    l2_cache_size_mb: int
    memory_bandwidth_gb_s: float

    # Register specs
    registers_per_sm: int
    max_registers_per_thread: int

    # Compute specs
    fp32_tflops: float
    fp16_tflops: float
    tensor_core_tflops: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "compute_capability": self.compute_capability,
            "num_sms": self.num_sms,
            "max_warps_per_sm": self.max_warps_per_sm,
            "max_threads_per_sm": self.max_threads_per_sm,
            "max_threads_per_block": self.max_threads_per_block,
            "warp_size": self.warp_size,
            "shared_memory_per_sm_kb": self.shared_memory_per_sm_kb,
            "l1_cache_size_kb": self.l1_cache_size_kb,
            "l2_cache_size_mb": self.l2_cache_size_mb,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "registers_per_sm": self.registers_per_sm,
            "max_registers_per_thread": self.max_registers_per_thread,
            "fp32_tflops": self.fp32_tflops,
            "fp16_tflops": self.fp16_tflops,
            "tensor_core_tflops": self.tensor_core_tflops,
        }


# Predefined GPU architectures
TESLA_V100 = GPUArchitecture(
    name="Tesla V100",
    compute_capability="7.0",
    num_sms=80,
    max_warps_per_sm=64,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=96,
    l1_cache_size_kb=128,
    l2_cache_size_mb=6,
    memory_bandwidth_gb_s=900,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=15.7,
    fp16_tflops=31.4,
    tensor_core_tflops=125,
)

A100 = GPUArchitecture(
    name="A100",
    compute_capability="8.0",
    num_sms=108,
    max_warps_per_sm=64,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=164,
    l1_cache_size_kb=192,
    l2_cache_size_mb=40,
    memory_bandwidth_gb_s=1555,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=19.5,
    fp16_tflops=312,
    tensor_core_tflops=312,
)

RTX_3090 = GPUArchitecture(
    name="RTX 3090",
    compute_capability="8.6",
    num_sms=82,
    max_warps_per_sm=48,
    max_threads_per_sm=1536,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=100,
    l1_cache_size_kb=128,
    l2_cache_size_mb=6,
    memory_bandwidth_gb_s=936,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=35.6,
    fp16_tflops=71,
    tensor_core_tflops=285,
)

# Architecture registry
ARCHITECTURES: Dict[str, GPUArchitecture] = {
    "cuda:70": TESLA_V100,
    "cuda:7.0": TESLA_V100,
    "Tesla V100": TESLA_V100,
    "cuda:80": A100,
    "cuda:8.0": A100,
    "A100": A100,
    "cuda:86": RTX_3090,
    "cuda:8.6": RTX_3090,
    "RTX 3090": RTX_3090,
}


def get_architecture(identifier: str) -> GPUArchitecture:
    """
    Get GPU architecture by identifier

    Args:
        identifier: Architecture identifier (e.g., "cuda:70", "Tesla V100")

    Returns:
        GPUArchitecture object

    Raises:
        KeyError: If architecture not found
    """
    if identifier in ARCHITECTURES:
        return ARCHITECTURES[identifier]

    # Default to V100
    return TESLA_V100
