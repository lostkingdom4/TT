"""
State management for LangGraph-based multi-agent GPU optimization system
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class KernelConfig:
    """Triton kernel configuration"""
    kernel_name: str
    ttgir: str  # Triton GPU IR
    ttir: Optional[str] = None  # High-level Triton IR
    source_code: Optional[str] = None  # Original Triton source

    # Current configuration
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    block_size: Dict[str, int] = field(default_factory=dict)  # {"XBLOCK": 256, ...}
    grid: Dict[str, int] = field(default_factory=dict)  # {"x": 1024, "y": 1, "z": 1}
    grid_formula: str = ""  # Grid lambda source, e.g. "lambda meta: (triton.cdiv(M, meta['BLOCK_M']), ...)"

    # Memory characteristics
    shared_memory_bytes: int = 0
    global_loads: int = 0
    global_stores: int = 0
    eviction_policies: List[str] = field(default_factory=list)

    # Tensor information
    tensor_shapes: Dict[str, List[int]] = field(default_factory=dict)
    tensor_dtypes: Dict[str, str] = field(default_factory=dict)

    # Actual constexpr parameter names from the kernel (extracted from shapes JSON kwargs)
    # e.g., ["BLOCK_SIZE"] for softmax, ["XBLOCK", "YBLOCK"] for 2D kernels
    constexpr_param_names: List[str] = field(default_factory=list)

    # Architecture
    target_architecture: str = "cuda:70"  # cuda:70 (V100), cuda:80 (A100), cuda:90 (H100)

    # Performance baseline
    baseline_time_ms: Optional[float] = None
    baseline_occupancy: Optional[float] = None

    # Optimization target
    optimization_target: Literal["throughput", "latency", "energy"] = "throughput"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def __repr__(self) -> str:
        return (f"KernelConfig(name={self.kernel_name}, "
                f"warps={self.num_warps}, stages={self.num_stages}, "
                f"block_size={self.block_size})")


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str  # Agent name
    recipient: str  # "coordinator" or specific agent
    message_type: Literal["analysis", "constraint", "proposal", "conflict"]
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1-10, lower is higher priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class OptimizationResult:
    """Final optimization result"""
    kernel_name: str
    original_config: Dict[str, Any]
    recommended_config: Dict[str, Any]
    expected_improvement: str  # e.g., "25%"
    confidence_score: float  # 0.0-1.0

    # Agent analyses
    warp_analysis: Dict[str, Any] = field(default_factory=dict)
    sm_analysis: Dict[str, Any] = field(default_factory=dict)
    gpu_analysis: Dict[str, Any] = field(default_factory=dict)
    coordinator_reasoning: str = ""

    # Trade-offs and explanations
    tradeoffs: List[str] = field(default_factory=list)
    implementation_instructions: str = ""

    # Performance predictions
    predicted_speedup: Optional[float] = None
    risk_assessment: str = "low"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class OptimizationState(TypedDict):
    """State passed through the LangGraph workflow"""

    # Input
    kernel_config: KernelConfig

    # Agent messages
    messages: List[AgentMessage]

    # Agent analyses
    warp_analysis: Optional[Dict[str, Any]]
    sm_analysis: Optional[Dict[str, Any]]
    gpu_analysis: Optional[Dict[str, Any]]

    # Constraints propagated from higher levels
    constraints: Dict[str, Any]  # e.g., {"max_shared_memory": 49152, "min_occupancy": 0.5}

    # Intermediate proposals (lists of proposals from each agent)
    warp_proposal: Optional[List[Dict[str, Any]]]
    sm_proposal: Optional[List[Dict[str, Any]]]
    gpu_proposal: Optional[List[Dict[str, Any]]]

    # Final result
    optimization_result: Optional[OptimizationResult]

    # Workflow control
    iteration: int
    max_iterations: int
    converged: bool
    error: Optional[str]

    # Iteration history and feedback
    iteration_history: List[Dict[str, Any]]  # Track previous iterations
    refinement_focus: Optional[str]  # "warp"|"sm"|"gpu" - which bottleneck to focus on
    previous_coordinator_feedback: Optional[str]  # Feedback for agents

    # Benchmark-in-the-loop fields (Approach 1: closed-loop verification)
    benchmark_results: Optional[Dict[str, Any]]  # Latest: {original_stats, recommended_stats, speedup, ...}
    baseline_benchmark_result: Optional[Dict[str, Any]]  # Cached original-config timing (computed once)
    regression_detected: bool  # True if latest recommendation was slower than original
    benchmark_iterations: int  # Count of benchmark verification rounds so far
    benchmark_config: Optional[Dict[str, Any]]  # {kernel_module, kernel_name, kernel_class, shapes_file, ...}

    # Multi-candidate search fields (Approach 5: LLM-guided sampling)
    candidate_configs: List[Dict[str, Any]]  # Coordinator's ranked candidates
        # Each: {"config": {...}, "rationale": "...", "confidence": float, "variant_type": str}
    benchmark_results_per_candidate: List[Dict[str, Any]]  # Per-candidate benchmark results
        # Each: {"candidate_index": int, "config": {...}, "stats": {...}, "speedup": float}
    benchmark_history: List[Dict[str, Any]]  # All completed benchmark rounds (for cross-round feedback)
        # Each: {"round": int, "results": [<benchmark_results_per_candidate snapshot>]}
    orchestrator_alternatives: List[Dict[str, Any]]  # Orchestrator's alternative configs for Phase 2
    benchmarked_configs: Dict[str, float]  # Global dedup: _cfg_key -> speedup

    # Approach 7: Negotiation-based agent architecture (2 domain agents + orchestrator)
    data_movement_proposal: Optional[Dict[str, Any]]    # DataMovementAgent output
    resource_budget_proposal: Optional[Dict[str, Any]]   # ResourceBudgetAgent output

    # Grid computation (programmatic, not LLM)
    computed_grid: Optional[Dict[str, Any]]              # grid dims from compute_grid()
    execution_metrics: Optional[Dict[str, Any]]          # wave quantization, SM saturation

    # Negotiation
    conflict_detection: Optional[Dict[str, Any]]         # Orchestrator's conflict analysis
    negotiation_history: Optional[List[Dict[str, Any]]]  # Negotiation turns (if multi-round)
    consensus_reached: bool                               # Whether agents agreed
    negotiation_round: int                                # Current negotiation round (0 = initial)
    max_negotiation_rounds: int                           # Max rounds before forcing proceed (default 2)
    negotiation_in_progress: bool                         # True when in negotiation sub-loop

    # Feedback loop
    agent_feedback: Optional[Dict[str, Any]]             # Per-agent scoped feedback
    feedback_iteration: int                               # Current feedback round (0, 1, 2)
    max_feedback_iterations: int                          # Default 2
    profiling_data: Optional[Dict[str, Any]]             # ncu output if available
