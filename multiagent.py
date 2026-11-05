"""
Multi-Agent System for Hardware-Aware Triton Kernel Tuning with Zero-Execution Analysis

This system uses multiple specialized agents to analyze and optimize Triton kernel configurations
without executing the kernels, using analytical models and hardware-aware heuristics.
"""

import re
import ast
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUArchitecture(Enum):
    """Supported GPU architectures"""
    NVIDIA_VOLTA = "volta"
    NVIDIA_AMPERE = "ampere"
    NVIDIA_ADA = "ada"
    NVIDIA_HOPPER = "hopper"
    INTEL_XE_HPG = "xe_hpg"
    INTEL_XE_HPC = "xe_hpc"
    AMD_RDNA3 = "rdna3"
    AMD_CDNA3 = "cdna3"

class GPUType(Enum):
    """Specific GPU models with detailed specifications"""
    # NVIDIA Volta GPUs
    V100_16GB = "v100_16gb"
    V100_32GB = "v100_32gb"

    # NVIDIA Ampere GPUs
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    L40S = "l40s"

    # Intel Xe-HPG GPUs
    INTEL_B580 = "intel_b580"

@dataclass
class KernelAnalysis:
    """Analysis results from kernel source code"""
    memory_ops: int
    arithmetic_ops: int
    control_flow_complexity: int
    shared_memory_usage: int
    register_pressure: int
    vectorization_potential: float
    memory_access_pattern: str  # "coalesced", "strided", "random"
    reduction_operations: int
    synchronization_points: int

@dataclass
class HardwareSpec:
    """Hardware specifications for a specific GPU"""
    gpu_type: GPUType
    architecture: GPUArchitecture
    model_name: str
    sm_count: int
    max_threads_per_sm: int
    max_threads_per_block: int
    max_shared_memory_per_block: int
    max_registers_per_thread: int
    max_registers_per_block: int
    memory_bandwidth_gb_s: float
    compute_capability_tflops: float
    l1_cache_size_kb: int
    l2_cache_size_mb: int
    memory_size_gb: int
    base_clock_mhz: int
    boost_clock_mhz: int
    warp_size: int = 32

@dataclass
class KernelConfig:
    """Triton kernel configuration"""
    block_sizes: Dict[str, int]  # e.g., {"XBLOCK": 128, "YBLOCK": 64}
    num_warps: int
    num_stages: int
    num_ctas: int = 1

@dataclass
class PerformanceMetrics:
    """Predicted performance metrics"""
    occupancy: float
    memory_efficiency: float
    compute_efficiency: float
    register_pressure_score: float
    predicted_latency_ms: float
    bandwidth_utilization: float
    overall_score: float

class Agent(ABC):
    """Base class for all agents in the multi-agent system"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")

    @abstractmethod
    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Analyze kernel configuration and return metrics"""
        pass

class RegisterPressureAnalyzer(Agent):
    """Agent for estimating register usage and spilling without execution"""

    def __init__(self):
        super().__init__("RegisterPressureAnalyzer")

    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Estimate register pressure based on static analysis"""

        # Base register usage estimation
        base_registers = 8  # Minimum overhead

        # Add registers for each memory operation (address calculation, data storage)
        memory_registers = min(kernel_analysis.memory_ops * 2, 32)

        # Add registers for arithmetic operations (intermediate results)
        arithmetic_registers = min(kernel_analysis.arithmetic_ops * 0.5, 24)

        # Control flow adds register pressure
        control_flow_registers = kernel_analysis.control_flow_complexity * 2

        # Block size affects register usage (larger blocks need more indexing)
        total_block_size = 1
        for block_size in config.block_sizes.values():
            total_block_size *= block_size

        block_size_factor = min(math.log2(total_block_size) * 0.5, 8)

        # Vectorization can reduce register pressure
        vectorization_reduction = kernel_analysis.vectorization_potential * 4

        estimated_registers = (base_registers + memory_registers + arithmetic_registers +
                             control_flow_registers + block_size_factor - vectorization_reduction)

        # Clamp to reasonable bounds
        estimated_registers = max(8, min(estimated_registers, 255))

        # Calculate spilling probability
        max_regs_per_thread = hardware_spec.max_registers_per_thread
        spill_probability = max(0, (estimated_registers - max_regs_per_thread) / max_regs_per_thread)

        # Calculate threads per block with register constraints
        threads_per_block = config.num_warps * hardware_spec.warp_size
        total_registers_needed = threads_per_block * estimated_registers

        register_limited_threads = hardware_spec.max_registers_per_block // estimated_registers
        register_occupancy = min(1.0, register_limited_threads / threads_per_block)

        return {
            "estimated_registers_per_thread": int(estimated_registers),
            "spill_probability": spill_probability,
            "register_occupancy": register_occupancy,
            "register_pressure_score": max(0, 1.0 - spill_probability * 2)
        }

class MemoryHierarchyOptimizer(Agent):
    """Agent for modeling shared memory and cache behavior"""

    def __init__(self):
        super().__init__("MemoryHierarchyOptimizer")

    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Analyze memory hierarchy efficiency"""

        # Shared memory usage analysis
        estimated_shared_mem = self._estimate_shared_memory_usage(kernel_source, config)
        shared_mem_efficiency = min(1.0, estimated_shared_mem / hardware_spec.max_shared_memory_per_block)

        # Cache efficiency based on access patterns
        cache_efficiency = self._estimate_cache_efficiency(kernel_analysis, config, hardware_spec)

        # Memory coalescing analysis
        coalescing_efficiency = self._analyze_memory_coalescing(kernel_analysis, config)

        # Memory bandwidth utilization
        bandwidth_utilization = self._estimate_bandwidth_utilization(
            kernel_analysis, config, hardware_spec)

        return {
            "shared_memory_usage_bytes": estimated_shared_mem,
            "shared_memory_efficiency": shared_mem_efficiency,
            "cache_efficiency": cache_efficiency,
            "coalescing_efficiency": coalescing_efficiency,
            "bandwidth_utilization": bandwidth_utilization,
            "memory_hierarchy_score": (cache_efficiency + coalescing_efficiency +
                                     bandwidth_utilization) / 3.0
        }

    def _estimate_shared_memory_usage(self, kernel_source: str, config: KernelConfig) -> int:
        """Estimate shared memory usage from kernel source"""
        # Look for shared memory allocations in the source
        shared_mem_pattern = r'tl\.zeros\(\[([^\]]+)\].*shared'
        shared_allocations = re.findall(shared_mem_pattern, kernel_source)

        total_shared_mem = 0
        for allocation in shared_allocations:
            # Parse allocation size
            try:
                # Simple heuristic: assume float32 (4 bytes) and evaluate block sizes
                size_expr = allocation.strip()
                for block_name, block_size in config.block_sizes.items():
                    size_expr = size_expr.replace(block_name, str(block_size))

                # Evaluate simple expressions
                if size_expr.isdigit():
                    total_shared_mem += int(size_expr) * 4  # 4 bytes per float32
                else:
                    # Fallback estimation
                    total_shared_mem += max(config.block_sizes.values()) * 4
            except:
                # Conservative estimate
                total_shared_mem += 1024

        return total_shared_mem

    def _estimate_cache_efficiency(self, kernel_analysis: KernelAnalysis,
                                 config: KernelConfig, hardware_spec: HardwareSpec) -> float:
        """Estimate L1/L2 cache efficiency"""
        # Cache efficiency depends on access patterns and block sizes
        if kernel_analysis.memory_access_pattern == "coalesced":
            base_efficiency = 0.9
        elif kernel_analysis.memory_access_pattern == "strided":
            base_efficiency = 0.6
        else:  # random
            base_efficiency = 0.3

        # Larger blocks can improve cache locality
        max_block_size = max(config.block_sizes.values())
        size_factor = min(1.0, max_block_size / 128.0)  # Optimal around 128

        return base_efficiency * (0.8 + 0.2 * size_factor)

    def _analyze_memory_coalescing(self, kernel_analysis: KernelAnalysis,
                                 config: KernelConfig) -> float:
        """Analyze memory access coalescing efficiency"""
        if kernel_analysis.memory_access_pattern == "coalesced":
            return 0.95
        elif kernel_analysis.memory_access_pattern == "strided":
            # Coalescing depends on stride and warp size
            return 0.7
        else:
            return 0.3

    def _estimate_bandwidth_utilization(self, kernel_analysis: KernelAnalysis,
                                      config: KernelConfig, hardware_spec: HardwareSpec) -> float:
        """Estimate memory bandwidth utilization"""
        # Calculate bytes per operation
        bytes_per_op = kernel_analysis.memory_ops * 4  # Assume float32

        # Operations per second estimation
        threads_per_block = config.num_warps * hardware_spec.warp_size
        blocks_per_sm = hardware_spec.max_threads_per_sm // threads_per_block
        total_throughput = (hardware_spec.sm_count * blocks_per_sm * threads_per_block *
                          kernel_analysis.arithmetic_ops)

        # Bandwidth requirement
        bandwidth_required = total_throughput * bytes_per_op / 1e9  # GB/s

        return min(1.0, bandwidth_required / hardware_spec.memory_bandwidth_gb_s)

class ArchitectureProfiler(Agent):
    """Agent for GPU-specific constraints and capabilities"""

    def __init__(self):
        super().__init__("ArchitectureProfiler")
        self.hardware_specs = self._initialize_hardware_specs()

    def _initialize_hardware_specs(self) -> Dict[GPUType, HardwareSpec]:
        """Initialize hardware specifications for specific GPU models"""
        return {
            # NVIDIA Volta GPUs
            GPUType.V100_16GB: HardwareSpec(
                gpu_type=GPUType.V100_16GB,
                architecture=GPUArchitecture.NVIDIA_VOLTA,
                model_name="NVIDIA Tesla V100 16GB",
                sm_count=80,
                max_threads_per_sm=2048,
                max_threads_per_block=1024,
                max_shared_memory_per_block=49152,
                max_registers_per_thread=255,
                max_registers_per_block=65536,
                memory_bandwidth_gb_s=900.0,
                compute_capability_tflops=125.0,
                l1_cache_size_kb=128,
                l2_cache_size_mb=6,
                memory_size_gb=16,
                base_clock_mhz=1245,
                boost_clock_mhz=1380
            ),
            GPUType.V100_32GB: HardwareSpec(
                gpu_type=GPUType.V100_32GB,
                architecture=GPUArchitecture.NVIDIA_VOLTA,
                model_name="NVIDIA Tesla V100 32GB",
                sm_count=80,
                max_threads_per_sm=2048,
                max_threads_per_block=1024,
                max_shared_memory_per_block=49152,
                max_registers_per_thread=255,
                max_registers_per_block=65536,
                memory_bandwidth_gb_s=900.0,
                compute_capability_tflops=125.0,
                l1_cache_size_kb=128,
                l2_cache_size_mb=6,
                memory_size_gb=32,
                base_clock_mhz=1245,
                boost_clock_mhz=1380
            ),
            # NVIDIA Ampere GPUs
            GPUType.A100_40GB: HardwareSpec(
                gpu_type=GPUType.A100_40GB,
                architecture=GPUArchitecture.NVIDIA_AMPERE,
                model_name="NVIDIA A100 40GB",
                sm_count=108,
                max_threads_per_sm=2048,
                max_threads_per_block=1024,
                max_shared_memory_per_block=49152,
                max_registers_per_thread=255,
                max_registers_per_block=65536,
                memory_bandwidth_gb_s=1555.0,
                compute_capability_tflops=312.0,
                l1_cache_size_kb=192,
                l2_cache_size_mb=40,
                memory_size_gb=40,
                base_clock_mhz=765,
                boost_clock_mhz=1410
            ),
            GPUType.A100_80GB: HardwareSpec(
                gpu_type=GPUType.A100_80GB,
                architecture=GPUArchitecture.NVIDIA_AMPERE,
                model_name="NVIDIA A100 80GB",
                sm_count=108,
                max_threads_per_sm=2048,
                max_threads_per_block=1024,
                max_shared_memory_per_block=49152,
                max_registers_per_thread=255,
                max_registers_per_block=65536,
                memory_bandwidth_gb_s=2039.0,
                compute_capability_tflops=312.0,
                l1_cache_size_kb=192,
                l2_cache_size_mb=40,
                memory_size_gb=80,
                base_clock_mhz=765,
                boost_clock_mhz=1410
            ),
            GPUType.L40S: HardwareSpec(
                gpu_type=GPUType.L40S,
                architecture=GPUArchitecture.NVIDIA_ADA,
                model_name="NVIDIA L40S",
                sm_count=142,
                max_threads_per_sm=1536,
                max_threads_per_block=1024,
                max_shared_memory_per_block=49152,
                max_registers_per_thread=255,
                max_registers_per_block=65536,
                memory_bandwidth_gb_s=864.0,
                compute_capability_tflops=91.6,
                l1_cache_size_kb=128,
                l2_cache_size_mb=72,
                memory_size_gb=48,
                base_clock_mhz=1110,
                boost_clock_mhz=2520
            ),
            # Intel Xe-HPG GPUs
            GPUType.INTEL_B580: HardwareSpec(
                gpu_type=GPUType.INTEL_B580,
                architecture=GPUArchitecture.INTEL_XE_HPG,
                model_name="Intel Arc B580",
                sm_count=20,  # Xe-cores
                max_threads_per_sm=1024,
                max_threads_per_block=512,
                max_shared_memory_per_block=32768,
                max_registers_per_thread=128,
                max_registers_per_block=32768,
                memory_bandwidth_gb_s=456.0,
                compute_capability_tflops=12.0,
                l1_cache_size_kb=64,
                l2_cache_size_mb=18,
                memory_size_gb=12,
                base_clock_mhz=2000,
                boost_clock_mhz=2670
            )
        }

    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Analyze architecture-specific constraints"""

        constraints_violated = []
        constraint_scores = {}

        # Thread block size constraints
        threads_per_block = config.num_warps * hardware_spec.warp_size
        if threads_per_block > hardware_spec.max_threads_per_block:
            constraints_violated.append("max_threads_per_block")
        constraint_scores["thread_block_size"] = min(1.0,
            hardware_spec.max_threads_per_block / threads_per_block)

        # Warp constraints (must be power of 2, reasonable range)
        if config.num_warps not in [1, 2, 4, 8, 16, 32]:
            constraints_violated.append("invalid_warp_count")
        constraint_scores["warp_configuration"] = 1.0 if config.num_warps in [2, 4, 8] else 0.8

        # Architecture-specific optimizations
        arch_score = self._get_architecture_specific_score(hardware_spec, config)
        constraint_scores["architecture_specific"] = arch_score

        return {
            "constraints_violated": constraints_violated,
            "constraint_scores": constraint_scores,
            "architecture_compatibility": len(constraints_violated) == 0,
            "architecture_score": sum(constraint_scores.values()) / len(constraint_scores)
        }

    def _get_architecture_specific_score(self, hardware_spec: HardwareSpec,
                                       config: KernelConfig) -> float:
        """Get architecture-specific optimization score"""
        if hardware_spec.architecture == GPUArchitecture.NVIDIA_AMPERE:
            # Ampere prefers certain warp counts and block sizes
            if config.num_warps in [4, 8]:
                return 1.0
            elif config.num_warps in [2, 16]:
                return 0.8
            else:
                return 0.6
        elif hardware_spec.architecture == GPUArchitecture.INTEL_XE_HPG:
            # Intel GPUs have different optimization preferences
            if config.num_warps in [2, 4]:
                return 1.0
            else:
                return 0.7
        else:
            return 0.8  # Default score for unknown architectures

class OccupancyCalculator(Agent):
    """Agent for theoretical occupancy modeling"""

    def __init__(self):
        super().__init__("OccupancyCalculator")

    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Calculate theoretical occupancy"""

        threads_per_block = config.num_warps * hardware_spec.warp_size

        # Thread limit
        max_blocks_per_sm_threads = hardware_spec.max_threads_per_sm // threads_per_block

        # Shared memory limit (estimated)
        estimated_shared_mem = kernel_analysis.shared_memory_usage
        if estimated_shared_mem > 0:
            max_blocks_per_sm_shmem = hardware_spec.max_shared_memory_per_block // estimated_shared_mem
        else:
            max_blocks_per_sm_shmem = float('inf')

        # Register limit (from RegisterPressureAnalyzer)
        estimated_registers = 32  # Default estimate, should be updated by RegisterPressureAnalyzer
        max_blocks_per_sm_regs = hardware_spec.max_registers_per_block // (threads_per_block * estimated_registers)

        # Actual occupancy is limited by the most restrictive resource
        max_blocks_per_sm = min(max_blocks_per_sm_threads, max_blocks_per_sm_shmem, max_blocks_per_sm_regs)

        # Theoretical maximum blocks per SM (architecture dependent)
        theoretical_max_blocks = 16  # Common limit for many architectures
        max_blocks_per_sm = min(max_blocks_per_sm, theoretical_max_blocks)

        occupancy = max_blocks_per_sm * threads_per_block / hardware_spec.max_threads_per_sm
        occupancy = min(1.0, occupancy)

        return {
            "occupancy": occupancy,
            "max_blocks_per_sm": max_blocks_per_sm,
            "threads_per_block": threads_per_block,
            "limiting_factor": self._get_limiting_factor(
                max_blocks_per_sm_threads, max_blocks_per_sm_shmem, max_blocks_per_sm_regs),
            "occupancy_score": occupancy
        }

    def _get_limiting_factor(self, thread_limit: int, shmem_limit: float, reg_limit: int) -> str:
        """Identify the limiting factor for occupancy"""
        min_limit = min(thread_limit, shmem_limit, reg_limit)

        if min_limit == thread_limit:
            return "threads"
        elif min_limit == shmem_limit:
            return "shared_memory"
        else:
            return "registers"

class PerformancePredictor(Agent):
    """Agent for surrogate model-based performance prediction"""

    def __init__(self):
        super().__init__("PerformancePredictor")

    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Predict performance using analytical models"""

        # Get basic metrics
        threads_per_block = config.num_warps * hardware_spec.warp_size

        # Compute efficiency model
        compute_efficiency = self._model_compute_efficiency(kernel_analysis, config, hardware_spec)

        # Memory efficiency model
        memory_efficiency = self._model_memory_efficiency(kernel_analysis, config, hardware_spec)

        # Latency prediction model
        predicted_latency = self._predict_latency(kernel_analysis, config, hardware_spec)

        # Overall performance score
        performance_score = (compute_efficiency * 0.4 + memory_efficiency * 0.6)

        return {
            "compute_efficiency": compute_efficiency,
            "memory_efficiency": memory_efficiency,
            "predicted_latency_ms": predicted_latency,
            "performance_score": performance_score,
            "throughput_estimate": 1.0 / max(predicted_latency, 0.001)  # Operations per ms
        }

    def _model_compute_efficiency(self, kernel_analysis: KernelAnalysis,
                                config: KernelConfig, hardware_spec: HardwareSpec) -> float:
        """Model compute efficiency"""
        # Base efficiency from arithmetic intensity
        arithmetic_intensity = kernel_analysis.arithmetic_ops / max(kernel_analysis.memory_ops, 1)
        base_efficiency = min(1.0, arithmetic_intensity / 10.0)  # Normalize to common range

        # Vectorization bonus
        vectorization_bonus = kernel_analysis.vectorization_potential * 0.2

        # Warp utilization
        warp_utilization = min(1.0, config.num_warps / 8.0)  # Optimal around 8 warps

        return min(1.0, base_efficiency + vectorization_bonus) * warp_utilization

    def _model_memory_efficiency(self, kernel_analysis: KernelAnalysis,
                               config: KernelConfig, hardware_spec: HardwareSpec) -> float:
        """Model memory efficiency"""
        # Base efficiency from access patterns
        if kernel_analysis.memory_access_pattern == "coalesced":
            base_efficiency = 0.9
        elif kernel_analysis.memory_access_pattern == "strided":
            base_efficiency = 0.6
        else:
            base_efficiency = 0.3

        # Block size impact on memory efficiency
        max_block_size = max(config.block_sizes.values()) if config.block_sizes else 128
        size_factor = 1.0 - abs(math.log2(max_block_size / 128.0)) * 0.1  # Optimal around 128
        size_factor = max(0.5, min(1.0, size_factor))

        return base_efficiency * size_factor

    def _predict_latency(self, kernel_analysis: KernelAnalysis,
                        config: KernelConfig, hardware_spec: HardwareSpec) -> float:
        """Predict kernel latency in milliseconds"""
        # Base latency from operation count
        base_ops = kernel_analysis.arithmetic_ops + kernel_analysis.memory_ops
        base_latency = base_ops / (hardware_spec.compute_capability_tflops * 1e12) * 1000  # ms

        # Memory latency component
        memory_latency = (kernel_analysis.memory_ops * 4) / (hardware_spec.memory_bandwidth_gb_s * 1e9) * 1000  # ms

        # Synchronization overhead
        sync_overhead = kernel_analysis.synchronization_points * 0.001  # 1μs per sync

        # Block size scaling factor
        threads_per_block = config.num_warps * hardware_spec.warp_size
        parallelism_factor = min(1.0, threads_per_block / 256.0)  # Optimal around 256 threads

        total_latency = (base_latency + memory_latency + sync_overhead) / parallelism_factor

        return max(0.001, total_latency)  # Minimum 1μs

class ConstraintValidator(Agent):
    """Agent for hardware limit validation"""

    def __init__(self):
        super().__init__("ConstraintValidator")

    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Validate hardware constraints"""

        violations = []
        warnings = []

        # Thread count validation
        threads_per_block = config.num_warps * hardware_spec.warp_size
        if threads_per_block > hardware_spec.max_threads_per_block:
            violations.append(f"Threads per block ({threads_per_block}) exceeds maximum ({hardware_spec.max_threads_per_block})")

        # Warp count validation
        if config.num_warps <= 0 or config.num_warps > 32:
            violations.append(f"Invalid warp count: {config.num_warps}")

        # Shared memory validation (estimated)
        if kernel_analysis.shared_memory_usage > hardware_spec.max_shared_memory_per_block:
            violations.append(f"Shared memory usage ({kernel_analysis.shared_memory_usage}) exceeds maximum ({hardware_spec.max_shared_memory_per_block})")

        # Block size validation
        for block_name, block_size in config.block_sizes.items():
            if block_size <= 0 or block_size > 2048:
                violations.append(f"Invalid block size {block_name}: {block_size}")
            elif not (block_size & (block_size - 1)) == 0:  # Check if power of 2
                warnings.append(f"Block size {block_name} ({block_size}) is not a power of 2")

        # Stages validation
        if config.num_stages <= 0 or config.num_stages > 8:
            violations.append(f"Invalid stage count: {config.num_stages}")

        is_valid = len(violations) == 0
        confidence = 1.0 - (len(violations) * 0.5 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)

        return {
            "is_valid": is_valid,
            "violations": violations,
            "warnings": warnings,
            "validation_score": confidence
        }

class KernelSourceAnalyzer:
    """Utility class for static analysis of Triton kernel source code"""

    @staticmethod
    def analyze_kernel(kernel_source: str) -> KernelAnalysis:
        """Perform static analysis of kernel source code"""

        # Count memory operations
        memory_ops = (len(re.findall(r'tl\.load', kernel_source)) +
                     len(re.findall(r'tl\.store', kernel_source)))

        # Count arithmetic operations
        arithmetic_patterns = [r'\+', r'-', r'\*', r'/', r'tl\.exp', r'tl\.log', r'tl\.sqrt']
        arithmetic_ops = sum(len(re.findall(pattern, kernel_source)) for pattern in arithmetic_patterns)

        # Control flow complexity
        control_flow_ops = (len(re.findall(r'if\s', kernel_source)) +
                          len(re.findall(r'for\s', kernel_source)) +
                          len(re.findall(r'while\s', kernel_source)))

        # Shared memory usage
        shared_memory_usage = len(re.findall(r'tl\.zeros.*shared', kernel_source)) * 1024  # Rough estimate

        # Register pressure estimation
        variable_count = len(set(re.findall(r'\b[a-zA-Z_]\w*\s*=', kernel_source)))
        register_pressure = min(variable_count * 2, 64)  # Rough estimate

        # Vectorization potential
        vector_ops = len(re.findall(r'tl\.\w+\(.*tl\.arange', kernel_source))
        vectorization_potential = min(1.0, vector_ops / max(arithmetic_ops, 1))

        # Memory access pattern analysis
        if 'tl.arange' in kernel_source and 'stride' not in kernel_source:
            memory_access_pattern = "coalesced"
        elif 'stride' in kernel_source:
            memory_access_pattern = "strided"
        else:
            memory_access_pattern = "random"

        # Reduction operations
        reduction_ops = (len(re.findall(r'tl\.sum', kernel_source)) +
                        len(re.findall(r'tl\.max', kernel_source)) +
                        len(re.findall(r'tl\.min', kernel_source)))

        # Synchronization points
        sync_points = len(re.findall(r'tl\.debug_barrier', kernel_source))

        return KernelAnalysis(
            memory_ops=memory_ops,
            arithmetic_ops=arithmetic_ops,
            control_flow_complexity=control_flow_ops,
            shared_memory_usage=shared_memory_usage,
            register_pressure=register_pressure,
            vectorization_potential=vectorization_potential,
            memory_access_pattern=memory_access_pattern,
            reduction_operations=reduction_ops,
            synchronization_points=sync_points
        )

class ConfigurationOrchestrator(Agent):
    """Coordinates multi-agent decisions and generates optimal configurations"""

    def __init__(self, hardware_spec: HardwareSpec):
        super().__init__("ConfigurationOrchestrator")
        self.hardware_spec = hardware_spec

        # Initialize all agents
        self.agents = {
            'register_pressure': RegisterPressureAnalyzer(),
            'memory_hierarchy': MemoryHierarchyOptimizer(),
            'architecture': ArchitectureProfiler(),
            'occupancy': OccupancyCalculator(),
            'performance': PerformancePredictor(),
            'constraints': ConstraintValidator()
        }

        self.analyzer = KernelSourceAnalyzer()

    def analyze(self, kernel_source: str, kernel_analysis: KernelAnalysis,
                hardware_spec: HardwareSpec, config: KernelConfig) -> Dict[str, Any]:
        """Coordinate analysis across all agents"""

        results = {}

        # Run all agent analyses
        for agent_name, agent in self.agents.items():
            try:
                agent_results = agent.analyze(kernel_source, kernel_analysis, hardware_spec, config)
                results[agent_name] = agent_results
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}")
                results[agent_name] = {"error": str(e)}

        # Aggregate scores
        scores = self._aggregate_scores(results)

        # Calculate overall configuration score
        overall_score = self._calculate_overall_score(scores)

        results['aggregated_scores'] = scores
        results['overall_score'] = overall_score

        return results

    def _aggregate_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate scores from all agents"""
        scores = {}

        # Extract scores from each agent
        if 'register_pressure' in results and 'register_pressure_score' in results['register_pressure']:
            scores['register_pressure'] = results['register_pressure']['register_pressure_score']

        if 'memory_hierarchy' in results and 'memory_hierarchy_score' in results['memory_hierarchy']:
            scores['memory_hierarchy'] = results['memory_hierarchy']['memory_hierarchy_score']

        if 'architecture' in results and 'architecture_score' in results['architecture']:
            scores['architecture'] = results['architecture']['architecture_score']

        if 'occupancy' in results and 'occupancy_score' in results['occupancy']:
            scores['occupancy'] = results['occupancy']['occupancy_score']

        if 'performance' in results and 'performance_score' in results['performance']:
            scores['performance'] = results['performance']['performance_score']

        if 'constraints' in results and 'validation_score' in results['constraints']:
            scores['constraints'] = results['constraints']['validation_score']

        return scores

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        weights = {
            'register_pressure': 0.15,
            'memory_hierarchy': 0.25,
            'architecture': 0.10,
            'occupancy': 0.20,
            'performance': 0.25,
            'constraints': 0.05
        }

        weighted_score = 0.0
        total_weight = 0.0

        for score_name, score_value in scores.items():
            if score_name in weights:
                weight = weights[score_name]
                weighted_score += score_value * weight
                total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def generate_configurations(self, kernel_source: str,
                              num_configs: int = 10) -> List[Tuple[KernelConfig, PerformanceMetrics]]:
        """Generate and rank optimal kernel configurations"""

        # Analyze kernel source
        kernel_analysis = self.analyzer.analyze_kernel(kernel_source)

        # Generate candidate configurations
        candidate_configs = self._generate_candidate_configs(kernel_analysis)

        # Evaluate each configuration
        config_results = []

        for config in candidate_configs:
            try:
                # Run multi-agent analysis
                analysis_results = self.analyze(kernel_source, kernel_analysis, self.hardware_spec, config)

                # Create performance metrics
                performance_metrics = self._create_performance_metrics(analysis_results)

                config_results.append((config, performance_metrics))

            except Exception as e:
                self.logger.error(f"Failed to analyze configuration {config}: {e}")
                continue

        # Sort by overall score
        config_results.sort(key=lambda x: x[1].overall_score, reverse=True)

        # Return top configurations
        return config_results[:num_configs]

    def _generate_candidate_configs(self, kernel_analysis: KernelAnalysis) -> List[KernelConfig]:
        """Generate candidate configurations based on kernel analysis"""

        configs = []

        # Block size candidates based on kernel characteristics
        if kernel_analysis.memory_ops > kernel_analysis.arithmetic_ops:
            # Memory-bound kernel - prefer smaller blocks for better memory efficiency
            block_sizes = [32, 64, 128]
        else:
            # Compute-bound kernel - can use larger blocks
            block_sizes = [64, 128, 256, 512]

        # Warp count candidates
        warp_counts = [1, 2, 4, 8, 16] if kernel_analysis.control_flow_complexity > 0 else [2, 4, 8]

        # Stage candidates
        stage_counts = [1, 2, 3] if kernel_analysis.memory_ops > 10 else [1, 2]

        # Generate all combinations
        for block_size in block_sizes:
            for num_warps in warp_counts:
                for num_stages in stage_counts:
                    # Check basic feasibility
                    threads_per_block = num_warps * self.hardware_spec.warp_size
                    if threads_per_block <= self.hardware_spec.max_threads_per_block:

                        # Generate block size configurations
                        if kernel_analysis.memory_access_pattern == "coalesced":
                            # 1D kernel
                            block_sizes_dict = {"XBLOCK": block_size}
                        else:
                            # Potentially 2D kernel
                            block_sizes_dict = {"XBLOCK": block_size, "YBLOCK": min(block_size, 64)}

                        config = KernelConfig(
                            block_sizes=block_sizes_dict,
                            num_warps=num_warps,
                            num_stages=num_stages
                        )
                        configs.append(config)

        return configs

    def _create_performance_metrics(self, analysis_results: Dict[str, Any]) -> PerformanceMetrics:
        """Create performance metrics from analysis results"""

        # Extract metrics from agent results
        occupancy = analysis_results.get('occupancy', {}).get('occupancy', 0.5)
        memory_efficiency = analysis_results.get('memory_hierarchy', {}).get('memory_hierarchy_score', 0.5)
        compute_efficiency = analysis_results.get('performance', {}).get('compute_efficiency', 0.5)
        register_pressure_score = analysis_results.get('register_pressure', {}).get('register_pressure_score', 0.5)
        predicted_latency = analysis_results.get('performance', {}).get('predicted_latency_ms', 1.0)
        bandwidth_utilization = analysis_results.get('memory_hierarchy', {}).get('bandwidth_utilization', 0.5)
        overall_score = analysis_results.get('overall_score', 0.5)

        return PerformanceMetrics(
            occupancy=occupancy,
            memory_efficiency=memory_efficiency,
            compute_efficiency=compute_efficiency,
            register_pressure_score=register_pressure_score,
            predicted_latency_ms=predicted_latency,
            bandwidth_utilization=bandwidth_utilization,
            overall_score=overall_score
        )

class MultiAgentTritonTuner:
    """Main interface for the multi-agent Triton kernel tuning system"""

    def __init__(self, gpu_type: GPUType = GPUType.A100_40GB, architecture: Optional[GPUArchitecture] = None):
        """
        Initialize multi-agent tuner for specific GPU type

        Args:
            gpu_type: Specific GPU model to optimize for
            architecture: Optional architecture override (auto-detected from gpu_type if not provided)
        """
        self.gpu_type = gpu_type
        self.hardware_spec = self._get_hardware_spec(gpu_type)
        self.architecture = architecture or self.hardware_spec.architecture
        self.orchestrator = ConfigurationOrchestrator(self.hardware_spec)

        logger.info(f"Initialized MultiAgentTritonTuner for {self.hardware_spec.model_name} ({self.architecture.value})")

    def _get_hardware_spec(self, gpu_type: GPUType) -> HardwareSpec:
        """Get hardware specification for the given GPU type"""
        # Initialize hardware specs (moved from ArchitectureProfiler for better organization)
        profiler = ArchitectureProfiler()
        specs = profiler.hardware_specs

        if gpu_type in specs:
            return specs[gpu_type]
        else:
            # Fallback to A100 if specific GPU not found
            logger.warning(f"GPU type {gpu_type.value} not found, falling back to A100 40GB")
            return specs[GPUType.A100_40GB]

    def tune_kernel(self, kernel_source: str, num_configs: int = 10) -> List[Tuple[KernelConfig, PerformanceMetrics]]:
        """
        Tune a Triton kernel using multi-agent analysis

        Args:
            kernel_source: Triton kernel source code
            num_configs: Number of top configurations to return

        Returns:
            List of (KernelConfig, PerformanceMetrics) tuples, sorted by performance
        """
        logger.info(f"Tuning kernel with {num_configs} configurations")
        logger.info(f"Target architecture: {self.architecture.value}")

        # Generate and evaluate configurations
        configurations = self.orchestrator.generate_configurations(kernel_source, num_configs)

        logger.info(f"Generated {len(configurations)} valid configurations")

        # Log top configuration
        if configurations:
            best_config, best_metrics = configurations[0]
            logger.info(f"Best configuration: {best_config}")
            logger.info(f"Best metrics: Overall score={best_metrics.overall_score:.3f}, "
                       f"Occupancy={best_metrics.occupancy:.3f}, "
                       f"Predicted latency={best_metrics.predicted_latency_ms:.3f}ms")

        return configurations

    def analyze_kernel_detailed(self, kernel_source: str, config: KernelConfig) -> Dict[str, Any]:
        """
        Perform detailed analysis of a specific kernel configuration

        Args:
            kernel_source: Triton kernel source code
            config: Kernel configuration to analyze

        Returns:
            Detailed analysis results from all agents
        """
        kernel_analysis = self.orchestrator.analyzer.analyze_kernel(kernel_source)
        return self.orchestrator.analyze(kernel_source, kernel_analysis, self.hardware_spec, config)

# Example usage and testing
def example_usage():
    """Example of how to use the multi-agent tuning system"""

    # Sample Triton kernel source
    sample_kernel = """
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    """

    # Initialize tuner for NVIDIA A100
    tuner = MultiAgentTritonTuner(GPUType.V100_32GB)

    # Tune the kernel
    top_configs = tuner.tune_kernel(sample_kernel, num_configs=5)

    # Print results
    print("Top 5 configurations:")
    for i, (config, metrics) in enumerate(top_configs):
        print(f"\n{i+1}. Configuration:")
        print(f"   Block sizes: {config.block_sizes}")
        print(f"   Warps: {config.num_warps}, Stages: {config.num_stages}")
        print(f"   Overall score: {metrics.overall_score:.3f}")
        print(f"   Occupancy: {metrics.occupancy:.3f}")
        print(f"   Predicted latency: {metrics.predicted_latency_ms:.3f}ms")

    # Detailed analysis of best configuration
    if top_configs:
        best_config = top_configs[0][0]
        detailed_analysis = tuner.analyze_kernel_detailed(sample_kernel, best_config)
        print(f"\nDetailed analysis of best configuration:")
        print(f"Register pressure: {detailed_analysis.get('register_pressure', {})}")
        print(f"Memory hierarchy: {detailed_analysis.get('memory_hierarchy', {})}")
        print(f"Architecture compatibility: {detailed_analysis.get('architecture', {})}")

if __name__ == "__main__":
    example_usage()
