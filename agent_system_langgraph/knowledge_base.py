"""
GPU Architecture Knowledge Base and Optimization Rules
"""

import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class GPUArchitecture:
    """GPU architecture specifications"""
    name: str
    compute_capability: str
    architecture_code: str  # e.g., "cuda:70"

    # SM (Streaming Multiprocessor) specs
    num_sms: int
    max_warps_per_sm: int
    max_threads_per_sm: int
    max_threads_per_block: int

    # Memory hierarchy
    shared_memory_per_sm_kb: int
    l1_cache_size_kb: int
    l2_cache_size_mb: int
    memory_bandwidth_gb_s: float

    # Register file
    registers_per_sm: int
    max_registers_per_thread: int

    # Compute capabilities
    fp32_tflops: float
    fp16_tflops: float

    # Fields with defaults must come after fields without defaults
    warp_size: int = 32
    tensor_core_tflops: Optional[float] = None

    # Per-component specs (for scoped agent prompts)
    warp_schedulers_per_sm: int = 4           # Number of warp schedulers per SM
    register_banks: int = 4                    # Number of register file banks
    l1_cache_line_bytes: int = 128            # L1 cache line size in bytes
    l2_cache_line_bytes: int = 128            # L2 cache line size in bytes
    memory_transaction_bytes: int = 32        # Global memory transaction size
    hbm_latency_cycles_approx: int = 400     # Approximate HBM latency in cycles

    # Optimization thresholds
    min_occupancy_target: float = 0.5
    coalescing_efficiency_threshold: float = 0.8
    bank_conflict_threshold: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context"""
        return {
            "name": self.name,
            "compute_capability": self.compute_capability,
            "sm_count": self.num_sms,
            "warp_size": self.warp_size,
            "max_warps_per_sm": self.max_warps_per_sm,
            "max_threads_per_sm": self.max_threads_per_sm,
            "shared_memory_per_sm_kb": self.shared_memory_per_sm_kb,
            "l1_cache_kb": self.l1_cache_size_kb,
            "l2_cache_mb": self.l2_cache_size_mb,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "fp32_tflops": self.fp32_tflops,
            "warp_schedulers_per_sm": self.warp_schedulers_per_sm,
            "register_banks": self.register_banks,
            "l1_cache_line_bytes": self.l1_cache_line_bytes,
            "l2_cache_line_bytes": self.l2_cache_line_bytes,
            "memory_transaction_bytes": self.memory_transaction_bytes,
            "hbm_latency_cycles_approx": self.hbm_latency_cycles_approx,
        }


# GPU Architecture Database
TESLA_V100 = GPUArchitecture(
    name="Tesla V100",
    compute_capability="7.0",
    architecture_code="cuda:70",
    num_sms=80,
    max_warps_per_sm=64,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=96,
    l1_cache_size_kb=128,
    l2_cache_size_mb=6,
    memory_bandwidth_gb_s=900.0,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=15.7,
    fp16_tflops=31.4,
    tensor_core_tflops=125.0,
)

A100 = GPUArchitecture(
    name="A100",
    compute_capability="8.0",
    architecture_code="cuda:80",
    num_sms=108,
    max_warps_per_sm=64,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=164,
    l1_cache_size_kb=192,
    l2_cache_size_mb=40,
    memory_bandwidth_gb_s=1555.0,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=19.5,
    fp16_tflops=312.0,
    tensor_core_tflops=312.0,
    hbm_latency_cycles_approx=350,
)

H100 = GPUArchitecture(
    name="H100",
    compute_capability="9.0",
    architecture_code="cuda:90",
    num_sms=132,
    max_warps_per_sm=64,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=228,
    l1_cache_size_kb=256,
    l2_cache_size_mb=50,
    memory_bandwidth_gb_s=3350.0,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=51.0,
    fp16_tflops=989.0,
    tensor_core_tflops=1979.0,
    hbm_latency_cycles_approx=300,
)

RTX_3090 = GPUArchitecture(
    name="RTX 3090",
    compute_capability="8.6",
    architecture_code="cuda:86",
    num_sms=82,
    max_warps_per_sm=48,
    max_threads_per_sm=1536,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=100,
    l1_cache_size_kb=128,
    l2_cache_size_mb=6,
    memory_bandwidth_gb_s=936.0,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=35.6,
    fp16_tflops=71.0,
    tensor_core_tflops=142.0,
    hbm_latency_cycles_approx=350,
)

A40 = GPUArchitecture(
    name="A40",
    compute_capability="8.6",
    architecture_code="cuda:a40",
    num_sms=84,
    max_warps_per_sm=48,
    max_threads_per_sm=1536,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=100,
    l1_cache_size_kb=128,
    l2_cache_size_mb=6,
    memory_bandwidth_gb_s=696.0,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=37.4,
    fp16_tflops=149.7,
    tensor_core_tflops=149.7,
    hbm_latency_cycles_approx=350,
)

L40S = GPUArchitecture(
    name="L40S",
    compute_capability="8.9",
    architecture_code="cuda:l40s",
    num_sms=142,
    max_warps_per_sm=48,
    max_threads_per_sm=1536,
    max_threads_per_block=1024,
    warp_size=32,
    shared_memory_per_sm_kb=100,
    l1_cache_size_kb=128,
    l2_cache_size_mb=96,
    memory_bandwidth_gb_s=864.0,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    fp32_tflops=91.6,
    fp16_tflops=366.3,
    tensor_core_tflops=366.3,
    hbm_latency_cycles_approx=300,
)

# Architecture registry
ARCHITECTURES: Dict[str, GPUArchitecture] = {
    "cuda:70": TESLA_V100,
    "cuda:80": A100,
    "cuda:86": RTX_3090,
    "cuda:a40": A40,
    "cuda:l40s": L40S,
    "cuda:90": H100,
}


def get_architecture(arch_code: str) -> GPUArchitecture:
    """Get GPU architecture by code"""
    if arch_code not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_code}. Available: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[arch_code]


# Optimization Rules Database
OPTIMIZATION_RULES = {
    "memory_bound": {
        "strategies": [
            "maximize_grid_parallelism",
            "improve_coalescing",
            "increase_l2_locality",
            "reduce_shared_memory_bank_conflicts",
        ],
        "metrics": ["memory_bandwidth_utilization", "coalescing_efficiency", "l2_hit_rate"],
        "priority": "latency_hiding",
    },
    "compute_bound": {
        "strategies": [
            "maximize_occupancy",
            "optimize_ilp",  # Instruction-level parallelism
            "reduce_register_pressure",
            "improve_warp_scheduling",
        ],
        "metrics": ["achieved_occupancy", "sm_utilization", "warp_stalls"],
        "priority": "throughput",
    },
    "balanced": {
        "strategies": [
            "balance_occupancy_and_parallelism",
            "optimize_cache_hierarchy",
            "tune_block_dimensions",
        ],
        "metrics": ["overall_throughput", "efficiency_ratio"],
        "priority": "balanced_optimization",
    },
}


# Kernel Type Heuristics
KERNEL_TYPE_PATTERNS = {
    "elementwise": {
        "indicators": ["tt.load", "tt.store", "few_compute_ops"],
        "characteristics": "memory_bound",
    },
    "reduction": {
        "indicators": ["tt.reduce", "atomic_ops"],
        "characteristics": "mixed",
    },
    "gemm": {
        "indicators": ["tt.dot", "mma_ops", "shared_memory_heavy"],
        "characteristics": "compute_bound",
    },
    "convolution": {
        "indicators": ["conv", "sliding_window"],
        "characteristics": "memory_bound",
    },
}


def detect_kernel_type(ttgir: str, kernel_name: str) -> str:
    """Detect kernel type from TTGIR and name"""
    ttgir_lower = ttgir.lower()
    name_lower = kernel_name.lower()

    # Check for GEMM/MatMul
    if "dot" in ttgir_lower or "mma" in ttgir_lower or "gemm" in name_lower or "matmul" in name_lower:
        return "gemm"

    # Check for reduction
    if "reduce" in ttgir_lower or "sum" in name_lower or "softmax" in name_lower:
        return "reduction"

    # Check for convolution
    if "conv" in name_lower:
        return "convolution"

    # Default to elementwise
    return "elementwise"


def get_optimization_strategy(kernel_type: str, architecture: GPUArchitecture) -> Dict[str, Any]:
    """Get optimization strategy for kernel type and architecture"""
    kernel_info = KERNEL_TYPE_PATTERNS.get(kernel_type, KERNEL_TYPE_PATTERNS["elementwise"])
    characteristic = kernel_info["characteristics"]
    rules = OPTIMIZATION_RULES.get(characteristic, OPTIMIZATION_RULES["balanced"])

    return {
        "kernel_type": kernel_type,
        "characteristic": characteristic,
        "strategies": rules["strategies"],
        "metrics": rules["metrics"],
        "architecture": architecture.name,
    }


def compute_grid(
    block_size: Dict[str, int],
    tensor_shapes: Dict[str, Any],
    kernel_type: str,
    num_sms: int,
) -> Dict[str, Any]:
    """Compute grid dimensions and execution quality metrics from block_size and tensor shapes.

    This is a pure function (no LLM call).  It replaces the GPUAgent's grid
    reasoning with deterministic computation.

    Args:
        block_size: Tile dimensions, e.g. ``{"BLOCK_SIZE": 4096}`` or
            ``{"BLOCK_M": 128, "BLOCK_N": 64}``.
        tensor_shapes: Tensor shapes dict, e.g.
            ``{"x": [4096, 256], "output": [4096, 256]}``.
        kernel_type: One of ``"elementwise"``, ``"reduction"``, ``"gemm"``,
            ``"convolution"``.
        num_sms: Number of SMs on the target GPU.

    Returns:
        Dict with ``"grid"`` (x/y/z), ``"total_ctas"``, ``"ctas_per_sm"``,
        ``"full_waves"``, ``"partial_wave_ctas"``,
        ``"partial_wave_utilization"``, ``"sm_saturation"``, and
        ``"wave_quantization_waste"``.
    """

    grid_x, grid_y, grid_z = 1, 1, 1

    if kernel_type == "reduction":
        # Reduction (e.g. softmax): one CTA per row.
        first_shape = next(
            (s for s in tensor_shapes.values() if isinstance(s, list) and len(s) >= 1),
            [1],
        )
        grid_x = first_shape[0]

    elif kernel_type == "gemm":
        # GEMM: grid_x = ceil(M / BLOCK_M), grid_y = ceil(N / BLOCK_N).
        shapes_2d = [
            s for s in tensor_shapes.values()
            if isinstance(s, list) and len(s) >= 2
        ]
        if len(shapes_2d) >= 2:
            M = shapes_2d[0][0]
            N = shapes_2d[1][1] if len(shapes_2d[1]) >= 2 else shapes_2d[1][0]
        elif len(shapes_2d) == 1:
            M = shapes_2d[0][0]
            N = shapes_2d[0][1]
        else:
            M, N = 1, 1

        block_m = block_size.get(
            "BLOCK_M",
            block_size.get("BLOCK_SIZE_M", next(iter(block_size.values()), 128)),
        )
        block_n = block_size.get(
            "BLOCK_N",
            block_size.get("BLOCK_SIZE_N", block_m),
        )
        grid_x = math.ceil(M / max(block_m, 1))
        grid_y = math.ceil(N / max(block_n, 1))

    else:
        # Elementwise / convolution / unknown: ceil(total_elements / block_size).
        max_elements = 0
        for shape in tensor_shapes.values():
            if isinstance(shape, list):
                elems = math.prod(shape) if shape else 0
            elif isinstance(shape, (int, float)):
                elems = int(shape)
            else:
                elems = 0
            max_elements = max(max_elements, elems)

        bs_val = next(iter(block_size.values()), 128)
        grid_x = math.ceil(max_elements / max(bs_val, 1)) if max_elements > 0 else 1

    # Ensure at least 1.
    grid_x = max(grid_x, 1)
    grid_y = max(grid_y, 1)
    grid_z = max(grid_z, 1)

    # --- Execution quality metrics ---
    total_ctas = grid_x * grid_y * grid_z
    safe_sms = max(num_sms, 1)
    ctas_per_sm = total_ctas / safe_sms
    full_waves = total_ctas // safe_sms
    partial_wave_ctas = total_ctas % safe_sms
    partial_wave_utilization = (
        partial_wave_ctas / safe_sms if partial_wave_ctas > 0 else 0.0
    )

    if ctas_per_sm >= 4:
        sm_saturation = "excellent"
    elif ctas_per_sm >= 2:
        sm_saturation = "good"
    elif ctas_per_sm >= 1:
        sm_saturation = "adequate"
    else:
        sm_saturation = "poor"

    wave_quantization_waste = (
        1.0 - partial_wave_utilization
        if total_ctas > 0 and partial_wave_ctas > 0
        else 0.0
    )

    return {
        "grid": {"x": grid_x, "y": grid_y, "z": grid_z},
        "total_ctas": total_ctas,
        "ctas_per_sm": ctas_per_sm,
        "full_waves": full_waves,
        "partial_wave_ctas": partial_wave_ctas,
        "partial_wave_utilization": partial_wave_utilization,
        "sm_saturation": sm_saturation,
        "wave_quantization_waste": wave_quantization_waste,
    }
