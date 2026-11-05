"""
Multi-Agent GPU Optimization System

A hierarchical multi-agent system for optimizing Triton GPU kernels.
Agents operate at different levels (thread, SM, GPU, coordinator) and
communicate via a message bus to propose and coordinate optimizations.
"""

from .kernel_config import (
    KernelConfig,
    GridDimensions,
    BlockSize,
    MemoryAccessPattern,
    PerformanceMetrics,
    TensorInfo,
    EvictionPolicy,
    KernelType,
)

from .messages import (
    BaseMessage,
    MessageType,
    MessagePriority,
    ResourceRequestMessage,
    ConstraintMessage,
    OptimizationProposal,
    ConflictResolutionMessage,
    AnalysisRequestMessage,
    AnalysisResponseMessage,
    StatusUpdateMessage,
)

from .message_bus import (
    MessageBus,
    AgentHierarchy,
)

from .architecture import (
    GPUArchitecture,
    TESLA_V100,
    A100,
    RTX_3090,
    ARCHITECTURES,
    get_architecture,
)

from .base_agent import BaseAgent

__version__ = "0.1.0"

__all__ = [
    # Kernel configuration
    "KernelConfig",
    "GridDimensions",
    "BlockSize",
    "MemoryAccessPattern",
    "PerformanceMetrics",
    "TensorInfo",
    "EvictionPolicy",
    "KernelType",

    # Messages
    "BaseMessage",
    "MessageType",
    "MessagePriority",
    "ResourceRequestMessage",
    "ConstraintMessage",
    "OptimizationProposal",
    "ConflictResolutionMessage",
    "AnalysisRequestMessage",
    "AnalysisResponseMessage",
    "StatusUpdateMessage",

    # Communication
    "MessageBus",
    "AgentHierarchy",

    # Architecture
    "GPUArchitecture",
    "TESLA_V100",
    "A100",
    "RTX_3090",
    "ARCHITECTURES",
    "get_architecture",

    # Base agent
    "BaseAgent",
]
