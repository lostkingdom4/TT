"""
Communication Bus for Multi-Agent System

Routes messages between agents based on type, priority, and agent hierarchy.
Maintains message queues and handles agent dependencies.
"""

import logging
from typing import Dict, List, Optional, Set
from queue import PriorityQueue
from threading import Lock
from collections import defaultdict
from datetime import datetime

from .messages import BaseMessage, MessageType, MessagePriority

log = logging.getLogger(__name__)


class AgentHierarchy:
    """Defines the hierarchy and dependencies between agent levels"""

    LEVELS = ['thread', 'sm', 'gpu', 'coordinator']

    @classmethod
    def get_parent_level(cls, level: str) -> Optional[str]:
        """Get the parent level in the hierarchy"""
        if level not in cls.LEVELS:
            return None
        idx = cls.LEVELS.index(level)
        if idx < len(cls.LEVELS) - 1:
            return cls.LEVELS[idx + 1]
        return None

    @classmethod
    def get_child_level(cls, level: str) -> Optional[str]:
        """Get the child level in the hierarchy"""
        if level not in cls.LEVELS:
            return None
        idx = cls.LEVELS.index(level)
        if idx > 0:
            return cls.LEVELS[idx - 1]
        return None

    @classmethod
    def can_communicate(cls, sender_level: str, recipient_level: str) -> bool:
        """Check if two agents at given levels can communicate directly"""
        if sender_level not in cls.LEVELS or recipient_level not in cls.LEVELS:
            return False

        sender_idx = cls.LEVELS.index(sender_level)
        recipient_idx = cls.LEVELS.index(recipient_level)

        # Can communicate with adjacent levels or coordinator can talk to all
        if sender_level == 'coordinator' or recipient_level == 'coordinator':
            return True

        return abs(sender_idx - recipient_idx) <= 1


class MessageBus:
    """
    Central communication bus for routing messages between agents

    Features:
    - Priority-based message queuing
    - Agent registration and discovery
    - Message routing based on hierarchy
    - Message history for debugging
    - Thread-safe operations
    """

    def __init__(self, max_history_size: int = 1000):
        self.agents: Dict[str, 'BaseAgent'] = {}  # agent_id -> agent
        self.agent_levels: Dict[str, str] = {}  # agent_id -> level
        self.message_queues: Dict[str, PriorityQueue] = defaultdict(PriorityQueue)
        self.message_history: List[BaseMessage] = []
        self.max_history_size = max_history_size
        self.lock = Lock()

        log.info("MessageBus initialized")

    def register_agent(self, agent: 'BaseAgent') -> None:
        """Register an agent with the bus"""
        with self.lock:
            if agent.agent_id in self.agents:
                log.warning(f"Agent {agent.agent_id} already registered, updating...")

            self.agents[agent.agent_id] = agent
            self.agent_levels[agent.agent_id] = agent.agent_level

            if agent.agent_id not in self.message_queues:
                self.message_queues[agent.agent_id] = PriorityQueue()

            log.info(f"Registered agent: {agent.agent_id} (level: {agent.agent_level})")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the bus"""
        with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                del self.agent_levels[agent_id]
                # Leave the queue for potential message delivery
                log.info(f"Unregistered agent: {agent_id}")

    def send_message(self, message: BaseMessage) -> bool:
        """
        Send a message from one agent to another

        Args:
            message: The message to send

        Returns:
            True if message was successfully queued, False otherwise
        """
        with self.lock:
            # Validate sender and recipient
            if message.sender_id not in self.agents:
                log.error(f"Unknown sender: {message.sender_id}")
                return False

            if message.recipient_id not in self.agents:
                log.error(f"Unknown recipient: {message.recipient_id}")
                return False

            # Check if communication is allowed
            sender_level = self.agent_levels[message.sender_id]
            recipient_level = self.agent_levels[message.recipient_id]

            if not AgentHierarchy.can_communicate(sender_level, recipient_level):
                log.warning(
                    f"Direct communication not allowed between {sender_level} "
                    f"and {recipient_level}, routing through hierarchy"
                )
                # Could implement routing through intermediate agents here
                return False

            # Add to recipient's queue
            self.message_queues[message.recipient_id].put((message.priority.value, message))

            # Add to history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history_size:
                self.message_history.pop(0)

            log.debug(
                f"Message {message.message_id} ({message.message_type.value}) "
                f"from {message.sender_id} to {message.recipient_id} "
                f"[priority: {message.priority.value}]"
            )

            return True

    def broadcast_message(self, sender_id: str, message: BaseMessage,
                         target_level: Optional[str] = None) -> int:
        """
        Broadcast a message to all agents at a specific level

        Args:
            sender_id: ID of the sending agent
            message: The message to broadcast
            target_level: Target agent level, or None for all agents

        Returns:
            Number of agents that received the message
        """
        count = 0
        message.sender_id = sender_id

        with self.lock:
            for agent_id, level in self.agent_levels.items():
                if agent_id == sender_id:
                    continue

                if target_level is None or level == target_level:
                    message.recipient_id = agent_id
                    message.message_id = f"{message.message_id}_{agent_id}"  # Unique per recipient
                    if self.send_message(message):
                        count += 1

        log.info(f"Broadcast from {sender_id} to {count} agents" +
                (f" at level {target_level}" if target_level else ""))
        return count

    def get_messages(self, agent_id: str, max_count: int = 10) -> List[BaseMessage]:
        """
        Retrieve messages for an agent

        Args:
            agent_id: ID of the agent
            max_count: Maximum number of messages to retrieve

        Returns:
            List of messages (highest priority first)
        """
        messages = []

        if agent_id not in self.message_queues:
            return messages

        queue = self.message_queues[agent_id]

        for _ in range(min(max_count, queue.qsize())):
            if not queue.empty():
                _, message = queue.get()
                messages.append(message)

        return messages

    def peek_message(self, agent_id: str) -> Optional[BaseMessage]:
        """
        Peek at the highest priority message without removing it

        Args:
            agent_id: ID of the agent

        Returns:
            The highest priority message or None
        """
        if agent_id not in self.message_queues:
            return None

        queue = self.message_queues[agent_id]

        if queue.empty():
            return None

        # Peek and put back
        priority, message = queue.get()
        queue.put((priority, message))
        return message

    def get_queue_size(self, agent_id: str) -> int:
        """Get the number of pending messages for an agent"""
        if agent_id not in self.message_queues:
            return 0
        return self.message_queues[agent_id].qsize()

    def get_agents_by_level(self, level: str) -> List[str]:
        """Get all agent IDs at a specific hierarchy level"""
        with self.lock:
            return [
                agent_id for agent_id, agent_level in self.agent_levels.items()
                if agent_level == level
            ]

    def get_message_history(self,
                           agent_id: Optional[str] = None,
                           message_type: Optional[MessageType] = None,
                           since: Optional[datetime] = None) -> List[BaseMessage]:
        """
        Get message history with optional filtering

        Args:
            agent_id: Filter by sender or recipient
            message_type: Filter by message type
            since: Filter by timestamp

        Returns:
            Filtered list of messages
        """
        filtered = self.message_history

        if agent_id:
            filtered = [
                m for m in filtered
                if m.sender_id == agent_id or m.recipient_id == agent_id
            ]

        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]

        if since:
            filtered = [m for m in filtered if m.timestamp >= since]

        return filtered

    def get_statistics(self) -> Dict[str, any]:
        """Get bus statistics"""
        with self.lock:
            total_pending = sum(q.qsize() for q in self.message_queues.values())

            return {
                "total_agents": len(self.agents),
                "agents_by_level": {
                    level: len(self.get_agents_by_level(level))
                    for level in AgentHierarchy.LEVELS
                },
                "total_pending_messages": total_pending,
                "pending_by_agent": {
                    agent_id: self.message_queues[agent_id].qsize()
                    for agent_id in self.agents
                },
                "total_history_messages": len(self.message_history),
            }

    def clear_queue(self, agent_id: str) -> int:
        """Clear all pending messages for an agent"""
        if agent_id not in self.message_queues:
            return 0

        count = self.message_queues[agent_id].qsize()
        self.message_queues[agent_id] = PriorityQueue()
        log.info(f"Cleared {count} messages for agent {agent_id}")
        return count

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (
            f"MessageBus: {stats['total_agents']} agents, "
            f"{stats['total_pending_messages']} pending messages"
        )
