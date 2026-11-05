"""
Message Protocol for Multi-Agent GPU Optimization

Defines message types for communication between optimization agents
at different hierarchy levels (thread, SM, GPU, coordinator).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import uuid


class MessageType(Enum):
    """Types of messages in the agent system"""
    RESOURCE_REQUEST = "resource_request"
    CONSTRAINT = "constraint"
    OPTIMIZATION_PROPOSAL = "optimization_proposal"
    CONFLICT_RESOLUTION = "conflict_resolution"
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    STATUS_UPDATE = "status_update"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0  # Immediate attention required
    HIGH = 1      # Important, process soon
    NORMAL = 2    # Standard priority
    LOW = 3       # Can be deferred


@dataclass
class BaseMessage:
    """Base class for all messages"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.STATUS_UPDATE
    sender_id: str = ""
    recipient_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'BaseMessage') -> bool:
        """Compare messages by priority and timestamp for queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "payload": self.payload,
        }


@dataclass
class ResourceRequestMessage(BaseMessage):
    """
    Request from lower-level agent to higher-level agent for resources

    Example: Thread agent requests more shared memory from SM agent
    """
    message_type: MessageType = MessageType.RESOURCE_REQUEST

    # Resource requirements
    requested_shared_memory_kb: Optional[float] = None
    requested_registers_per_thread: Optional[int] = None
    requested_warps: Optional[int] = None
    requested_ctas: Optional[int] = None

    # Justification
    reason: str = ""
    expected_benefit: Optional[str] = None  # e.g., "20% speedup"

    def __post_init__(self):
        """Populate payload from fields"""
        self.payload = {
            "requested_shared_memory_kb": self.requested_shared_memory_kb,
            "requested_registers_per_thread": self.requested_registers_per_thread,
            "requested_warps": self.requested_warps,
            "requested_ctas": self.requested_ctas,
            "reason": self.reason,
            "expected_benefit": self.expected_benefit,
        }


@dataclass
class ConstraintMessage(BaseMessage):
    """
    Constraints imposed by higher-level agent on lower-level agent

    Example: SM agent tells thread agent maximum shared memory available
    """
    message_type: MessageType = MessageType.CONSTRAINT

    # Resource constraints
    max_shared_memory_kb: Optional[float] = None
    max_registers_per_thread: Optional[int] = None
    max_warps_per_sm: Optional[int] = None
    max_ctas: Optional[int] = None
    max_threads_per_sm: Optional[int] = None

    # Architectural constraints
    compute_capability: str = "7.0"
    l1_cache_size_kb: Optional[int] = None
    l2_cache_size_mb: Optional[int] = None
    memory_bandwidth_gb_s: Optional[float] = None

    # Policy constraints
    eviction_policy_allowed: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Populate payload from fields"""
        self.payload = {
            "max_shared_memory_kb": self.max_shared_memory_kb,
            "max_registers_per_thread": self.max_registers_per_thread,
            "max_warps_per_sm": self.max_warps_per_sm,
            "max_ctas": self.max_ctas,
            "max_threads_per_sm": self.max_threads_per_sm,
            "compute_capability": self.compute_capability,
            "l1_cache_size_kb": self.l1_cache_size_kb,
            "l2_cache_size_mb": self.l2_cache_size_mb,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "eviction_policy_allowed": self.eviction_policy_allowed,
        }


@dataclass
class OptimizationProposal(BaseMessage):
    """
    Optimization suggestion from an agent

    Contains specific configuration changes and expected impact
    """
    message_type: MessageType = MessageType.OPTIMIZATION_PROPOSAL

    # Proposed configuration changes
    config_changes: Dict[str, Any] = field(default_factory=dict)

    # Analysis and justification
    analysis_summary: str = ""
    bottleneck_identified: Optional[str] = None
    optimization_strategy: str = ""

    # Expected impact
    expected_speedup: Optional[float] = None
    confidence_score: float = 0.5  # 0.0 to 1.0
    risk_level: str = "medium"  # low/medium/high

    # Trade-offs
    trade_offs: List[str] = field(default_factory=list)

    # Supporting data
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    predicted_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Populate payload from fields"""
        self.payload = {
            "config_changes": self.config_changes,
            "analysis_summary": self.analysis_summary,
            "bottleneck_identified": self.bottleneck_identified,
            "optimization_strategy": self.optimization_strategy,
            "expected_speedup": self.expected_speedup,
            "confidence_score": self.confidence_score,
            "risk_level": self.risk_level,
            "trade_offs": self.trade_offs,
            "baseline_metrics": self.baseline_metrics,
            "predicted_metrics": self.predicted_metrics,
        }


@dataclass
class ConflictResolutionMessage(BaseMessage):
    """
    Resolution from coordinator when multiple agents propose conflicting optimizations

    Example: When thread agent wants more shared memory but SM agent has none available
    """
    message_type: MessageType = MessageType.CONFLICT_RESOLUTION

    # Conflicting proposals
    conflicting_proposal_ids: List[str] = field(default_factory=list)

    # Resolution decision
    decision: str = ""  # "accept", "reject", "merge", "defer"
    selected_proposal_id: Optional[str] = None
    merged_config: Optional[Dict[str, Any]] = None

    # Reasoning
    resolution_reasoning: str = ""
    affected_agents: List[str] = field(default_factory=list)

    # Compromise details
    compromises_required: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Populate payload from fields"""
        self.payload = {
            "conflicting_proposal_ids": self.conflicting_proposal_ids,
            "decision": self.decision,
            "selected_proposal_id": self.selected_proposal_id,
            "merged_config": self.merged_config,
            "resolution_reasoning": self.resolution_reasoning,
            "affected_agents": self.affected_agents,
            "compromises_required": self.compromises_required,
        }


@dataclass
class AnalysisRequestMessage(BaseMessage):
    """Request for analysis from another agent"""
    message_type: MessageType = MessageType.ANALYSIS_REQUEST

    analysis_type: str = ""  # "memory", "occupancy", "coalescing", etc.
    kernel_config_data: Dict[str, Any] = field(default_factory=dict)
    specific_questions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Populate payload from fields"""
        self.payload = {
            "analysis_type": self.analysis_type,
            "kernel_config_data": self.kernel_config_data,
            "specific_questions": self.specific_questions,
        }


@dataclass
class AnalysisResponseMessage(BaseMessage):
    """Response to an analysis request"""
    message_type: MessageType = MessageType.ANALYSIS_RESPONSE

    request_id: str = ""
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Populate payload from fields"""
        self.payload = {
            "request_id": self.request_id,
            "analysis_results": self.analysis_results,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }


@dataclass
class StatusUpdateMessage(BaseMessage):
    """General status update from an agent"""
    message_type: MessageType = MessageType.STATUS_UPDATE

    status: str = "idle"  # idle, analyzing, proposing, waiting
    current_task: Optional[str] = None
    progress_percentage: Optional[float] = None
    issues: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Populate payload from fields"""
        self.payload = {
            "status": self.status,
            "current_task": self.current_task,
            "progress_percentage": self.progress_percentage,
            "issues": self.issues,
        }
