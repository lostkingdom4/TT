"""
Base Agent Abstract Class

Defines the interface for all optimization agents in the multi-agent system.
Agents operate at different hierarchy levels (thread, SM, GPU, coordinator)
and communicate via the message bus.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from .kernel_config import KernelConfig
from .messages import BaseMessage, OptimizationProposal
from .architecture import GPUArchitecture, get_architecture

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all optimization agents

    Agents analyze kernel configurations and propose optimizations
    within their scope of responsibility (thread, SM, GPU, or coordinator level).
    """

    def __init__(self,
                 agent_id: str,
                 agent_level: str,
                 target_architecture: str = "cuda:70",
                 message_bus: Optional['MessageBus'] = None):
        """
        Initialize the agent

        Args:
            agent_id: Unique identifier for this agent
            agent_level: Hierarchy level ('thread', 'sm', 'gpu', 'coordinator')
            target_architecture: Target GPU architecture
            message_bus: Communication bus for inter-agent messages
        """
        if agent_level not in ['thread', 'sm', 'gpu', 'coordinator']:
            raise ValueError(
                f"Invalid agent_level: {agent_level}. "
                "Must be one of: thread, sm, gpu, coordinator"
            )

        self.agent_id = agent_id
        self.agent_level = agent_level
        self.target_architecture = target_architecture
        self.knowledge_base: GPUArchitecture = get_architecture(target_architecture)
        self.message_bus = message_bus

        # Agent state
        self.active = False
        self.current_kernel_config: Optional[KernelConfig] = None
        self.analysis_results: Dict[str, Any] = {}
        self.pending_proposals: List[OptimizationProposal] = []

        log.info(
            f"Initialized {self.__class__.__name__} "
            f"(id: {agent_id}, level: {agent_level}, "
            f"arch: {self.knowledge_base.name})"
        )

        # Register with message bus if provided
        if self.message_bus:
            self.message_bus.register_agent(self)

    @abstractmethod
    def analyze(self, kernel_config: KernelConfig) -> Dict[str, Any]:
        """
        Analyze a kernel configuration and identify optimization opportunities

        Args:
            kernel_config: The kernel configuration to analyze

        Returns:
            Dictionary containing analysis results specific to this agent's level

        Example return:
            {
                "bottleneck": "memory_bandwidth",
                "utilization": 0.65,
                "recommendations": ["increase_block_size", "adjust_num_warps"],
                "confidence": 0.85
            }
        """
        pass

    @abstractmethod
    def suggest_optimizations(self,
                             analysis_results: Dict[str, Any]) -> List[OptimizationProposal]:
        """
        Generate optimization proposals based on analysis results

        Args:
            analysis_results: Results from the analyze() method

        Returns:
            List of optimization proposals with specific configuration changes
        """
        pass

    def receive_message(self, message: BaseMessage) -> None:
        """
        Receive and process a message from another agent

        Args:
            message: The message to process
        """
        log.debug(
            f"{self.agent_id} received message {message.message_id} "
            f"from {message.sender_id} (type: {message.message_type.value})"
        )

        # Process based on message type
        self._handle_message(message)

    def send_message(self, message: BaseMessage) -> bool:
        """
        Send a message to another agent via the message bus

        Args:
            message: The message to send

        Returns:
            True if message was sent successfully
        """
        if not self.message_bus:
            log.error(f"{self.agent_id}: No message bus configured")
            return False

        message.sender_id = self.agent_id
        return self.message_bus.send_message(message)

    def _handle_message(self, message: BaseMessage) -> None:
        """
        Internal message handler - can be overridden by subclasses

        Args:
            message: The message to handle
        """
        from .messages import MessageType

        if message.message_type == MessageType.ANALYSIS_REQUEST:
            self._handle_analysis_request(message)
        elif message.message_type == MessageType.CONSTRAINT:
            self._handle_constraint_message(message)
        elif message.message_type == MessageType.RESOURCE_REQUEST:
            self._handle_resource_request(message)
        elif message.message_type == MessageType.CONFLICT_RESOLUTION:
            self._handle_conflict_resolution(message)
        else:
            log.debug(f"{self.agent_id}: Unhandled message type {message.message_type}")

    def _handle_analysis_request(self, message: BaseMessage) -> None:
        """Handle analysis request messages"""
        log.info(f"{self.agent_id}: Received analysis request")
        # Subclasses can override

    def _handle_constraint_message(self, message: BaseMessage) -> None:
        """Handle constraint messages from higher-level agents"""
        log.info(f"{self.agent_id}: Received constraints")
        # Subclasses can override

    def _handle_resource_request(self, message: BaseMessage) -> None:
        """Handle resource requests from lower-level agents"""
        log.info(f"{self.agent_id}: Received resource request")
        # Subclasses can override

    def _handle_conflict_resolution(self, message: BaseMessage) -> None:
        """Handle conflict resolution from coordinator"""
        log.info(f"{self.agent_id}: Received conflict resolution")
        # Subclasses can override

    def process_pending_messages(self, max_messages: int = 10) -> int:
        """
        Process pending messages from the message bus

        Args:
            max_messages: Maximum number of messages to process

        Returns:
            Number of messages processed
        """
        if not self.message_bus:
            return 0

        messages = self.message_bus.get_messages(self.agent_id, max_messages)

        for message in messages:
            self.receive_message(message)

        return len(messages)

    def activate(self) -> None:
        """Activate this agent"""
        self.active = True
        log.info(f"{self.agent_id}: Activated")

    def deactivate(self) -> None:
        """Deactivate this agent"""
        self.active = False
        log.info(f"{self.agent_id}: Deactivated")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status

        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_id": self.agent_id,
            "agent_level": self.agent_level,
            "active": self.active,
            "architecture": self.knowledge_base.name,
            "pending_proposals": len(self.pending_proposals),
            "current_kernel": (
                self.current_kernel_config.kernel_name
                if self.current_kernel_config else None
            ),
            "message_queue_size": (
                self.message_bus.get_queue_size(self.agent_id)
                if self.message_bus else 0
            ),
        }

    def __str__(self) -> str:
        status = "active" if self.active else "inactive"
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"level={self.agent_level}, "
            f"status={status})"
        )

    def __repr__(self) -> str:
        return self.__str__()
