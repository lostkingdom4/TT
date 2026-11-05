"""
Kernel Configuration Data Structures

Defines the data structures for representing Triton kernel configurations,
performance metrics, and optimization parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class EvictionPolicy(Enum):
    """Cache eviction policies for memory operations"""
    EVICT_FIRST = "evict_first"
    EVICT_LAST = "evict_last"
    EVICT_NORMAL = "normal"


class KernelType(Enum):
    """Types of kernel operations"""
    ELEMENTWISE_1D = "1d"
    ELEMENTWISE_2D = "2d"
    REDUCTION = "reduction"
    REDUCTION_FIXED = "reduction_fixed"
    MATMUL = "matmul"
    CONV = "conv"
    UNKNOWN = "unknown"


@dataclass
class GridDimensions:
    """Grid dimensions for kernel launch"""
    x: int
    y: int = 1
    z: int = 1

    @property
    def total_ctas(self) -> int:
        """Total number of CTAs (cooperative thread arrays) in the grid"""
        return self.x * self.y * self.z

    def __str__(self) -> str:
        if self.z > 1:
            return f"[{self.x}, {self.y}, {self.z}]"
        elif self.y > 1:
            return f"[{self.x}, {self.y}]"
        else:
            return f"[{self.x}]"


@dataclass
class BlockSize:
    """Block size configuration for different kernel types"""
    XBLOCK: int
    YBLOCK: Optional[int] = None
    RBLOCK: Optional[int] = None  # Reduction block size

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary, excluding None values"""
        result = {"XBLOCK": self.XBLOCK}
        if self.YBLOCK is not None:
            result["YBLOCK"] = self.YBLOCK
        if self.RBLOCK is not None:
            result["RBLOCK"] = self.RBLOCK
        return result

    def __str__(self) -> str:
        parts = [f"XBLOCK={self.XBLOCK}"]
        if self.YBLOCK is not None:
            parts.append(f"YBLOCK={self.YBLOCK}")
        if self.RBLOCK is not None:
            parts.append(f"RBLOCK={self.RBLOCK}")
        return ", ".join(parts)


@dataclass
class MemoryAccessPattern:
    """Memory access characteristics"""
    global_loads: int = 0
    global_stores: int = 0
    shared_memory_bytes: int = 0
    coalescing_efficiency: str = "unknown"  # high/medium/low
    stride_pattern: Optional[str] = None
    eviction_policies: List[str] = field(default_factory=list)

    @property
    def total_memory_ops(self) -> int:
        """Total memory operations"""
        return self.global_loads + self.global_stores


@dataclass
class PerformanceMetrics:
    """Performance measurements and predictions"""
    execution_time_ms: Optional[float] = None
    execution_time_std: Optional[float] = None
    throughput_gb_s: Optional[float] = None
    compute_intensity: Optional[float] = None  # FLOPs per byte
    occupancy_percentage: Optional[float] = None
    l1_hit_rate: Optional[float] = None
    l2_hit_rate: Optional[float] = None
    memory_bandwidth_utilization: Optional[float] = None

    # Predicted metrics from analysis
    predicted_execution_time_ms: Optional[float] = None
    predicted_speedup: Optional[float] = None
    confidence_score: Optional[float] = None

    def __str__(self) -> str:
        if self.execution_time_ms is not None:
            return f"{self.execution_time_ms:.4f} ms"
        elif self.predicted_execution_time_ms is not None:
            return f"~{self.predicted_execution_time_ms:.4f} ms (predicted)"
        return "No metrics"


@dataclass
class TensorInfo:
    """Information about tensor operands"""
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int

    @property
    def numel(self) -> int:
        """Number of elements"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def __str__(self) -> str:
        return f"{self.shape} ({self.dtype})"


@dataclass
class KernelConfig:
    """
    Complete kernel configuration and metadata

    Represents a Triton GPU kernel configuration with all relevant
    parameters for optimization analysis.
    """
    # Identification
    kernel_name: str
    kernel_type: KernelType = KernelType.UNKNOWN

    # Thread/Block configuration
    num_warps: int = 4
    num_ctas: int = 1
    threads_per_warp: int = 32
    block_size: Optional[BlockSize] = None

    # Pipeline configuration
    num_stages: int = 2

    # Grid configuration
    grid: Optional[GridDimensions] = None

    # Memory configuration
    memory_access: MemoryAccessPattern = field(default_factory=MemoryAccessPattern)
    eviction_policy: EvictionPolicy = EvictionPolicy.EVICT_NORMAL

    # Tensor information
    input_tensors: List[TensorInfo] = field(default_factory=list)
    output_tensors: List[TensorInfo] = field(default_factory=list)

    # Performance
    current_performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Target architecture
    target_architecture: str = "cuda:70"  # Tesla V100 by default
    compute_capability: str = "7.0"

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_threads_per_cta(self) -> int:
        """Total threads per CTA (Cooperative Thread Array)"""
        return self.num_warps * self.threads_per_warp

    @property
    def total_threads_in_grid(self) -> int:
        """Total threads across entire grid"""
        if self.grid:
            return self.total_threads_per_cta * self.grid.total_ctas
        return self.total_threads_per_cta

    @property
    def shared_memory_per_cta_kb(self) -> float:
        """Shared memory per CTA in KB"""
        return self.memory_access.shared_memory_bytes / 1024.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "kernel_name": self.kernel_name,
            "kernel_type": self.kernel_type.value,
            "num_warps": self.num_warps,
            "num_ctas": self.num_ctas,
            "threads_per_warp": self.threads_per_warp,
            "block_size": self.block_size.to_dict() if self.block_size else None,
            "num_stages": self.num_stages,
            "grid": str(self.grid) if self.grid else None,
            "eviction_policy": self.eviction_policy.value,
            "target_architecture": self.target_architecture,
            "compute_capability": self.compute_capability,
            "memory_access": {
                "global_loads": self.memory_access.global_loads,
                "global_stores": self.memory_access.global_stores,
                "shared_memory_bytes": self.memory_access.shared_memory_bytes,
                "coalescing_efficiency": self.memory_access.coalescing_efficiency,
            },
            "current_performance": {
                "execution_time_ms": self.current_performance.execution_time_ms,
                "throughput_gb_s": self.current_performance.throughput_gb_s,
                "occupancy_percentage": self.current_performance.occupancy_percentage,
            },
        }

    def __str__(self) -> str:
        lines = [
            f"Kernel: {self.kernel_name}",
            f"  Type: {self.kernel_type.value}",
            f"  Warps: {self.num_warps}, CTAs: {self.num_ctas}",
        ]
        if self.block_size:
            lines.append(f"  Block size: {self.block_size}")
        if self.grid:
            lines.append(f"  Grid: {self.grid} ({self.grid.total_ctas} CTAs)")
        lines.append(f"  Shared mem: {self.shared_memory_per_cta_kb:.2f} KB/CTA")
        lines.append(f"  Performance: {self.current_performance}")
        return "\n".join(lines)
