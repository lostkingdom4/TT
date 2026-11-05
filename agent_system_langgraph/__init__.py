"""
LangGraph-based Multi-Agent GPU Kernel Optimization System

Two domain agents (DataMovement, ResourceBudget) analyze Triton GPU kernels
at complementary hardware levels, a programmatic grid computation replaces
LLM-based grid reasoning, and an Orchestrator agent resolves conflicts and
emits ranked candidate configurations verified through closed-loop benchmarking
with per-agent feedback loops. Legacy agents (Warp, SM, GPU, Coordinator)
remain available for backward compatibility.

Quick Start:
    >>> from agent_system_langgraph import optimize_kernel
    >>>
    >>> result = optimize_kernel(
    ...     kernel_name="my_kernel",
    ...     ttgir=ttgir_content,
    ...     current_config={"num_warps": 4, "block_size": {"XBLOCK": 1024}},
    ...     architecture="cuda:70",
    ...     llm_provider="deepseek",
    ... )
    >>>
    >>> print(result["result"]["recommended_config"])

Features:
    - Approach 7 architecture: DataMovement + ResourceBudget + Orchestrator + compute_grid()
    - Negotiation protocol with conflict detection and resolution
    - Per-agent scoped feedback from benchmark results
    - Closed-loop benchmarking with multi-candidate search
    - LLM-powered optimization (GPT-4, DeepSeek, Ollama, vLLM)
    - Architecture-aware (V100, A100, H100, RTX 3090)
    - Optional ncu profiling integration for precise feedback

See README.md for detailed documentation.
See QUICKSTART.md for setup and examples.
"""

from .state import OptimizationState, AgentMessage, KernelConfig, OptimizationResult
from .agents import WarpAgent, SMAgent, GPUAgent, CoordinatorAgent
from .agents import DataMovementAgent, ResourceBudgetAgent, OrchestratorAgent
from .orchestrator import GPUOptimizationOrchestrator, optimize_kernel
from .knowledge_base import get_architecture, ARCHITECTURES
from .llm_provider import LLMProvider

__version__ = "0.2.0"

__all__ = [
    # State
    "OptimizationState",
    "AgentMessage",
    "KernelConfig",
    "OptimizationResult",
    # Agents
    "WarpAgent",
    "SMAgent",
    "GPUAgent",
    "CoordinatorAgent",
    # Approach 7 Agents
    "DataMovementAgent",
    "ResourceBudgetAgent",
    "OrchestratorAgent",
    # Orchestrator
    "GPUOptimizationOrchestrator",
    "optimize_kernel",
    # Knowledge Base
    "get_architecture",
    "ARCHITECTURES",
    # LLM
    "LLMProvider",
]
