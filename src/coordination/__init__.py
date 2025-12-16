"""
Coordination Module for Multi-Agent System

This module provides coordination utilities for managing interactions between multiple agents,
including negotiation, conflict resolution, and centralized coordination.
"""

from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class NegotiationState(Enum):
    """Enumeration for negotiation states"""
    INITIATING = "initiating"
    PROPOSING = "proposing"
    ACCEPTING = "accepting"
    REJECTING = "rejecting"
    RESOLVED = "resolved"
    FAILED = "failed"


class ConflictType(Enum):
    """Enumeration for types of conflicts"""
    RESOURCE_CONFLICT = "resource_conflict"
    SCHEDULING_CONFLICT = "scheduling_conflict"
    PRIORITY_CONFLICT = "priority_conflict"
    GOAL_CONFLICT = "goal_conflict"
    COMMUNICATION_CONFLICT = "communication_conflict"


@dataclass
class Proposal:
    """Represents a negotiation proposal between agents"""
    proposer_id: str
    proposal_id: str
    content: Dict[str, Any]
    timestamp: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Proposal(proposer_id={self.proposer_id}, proposal_id={self.proposal_id}, priority={self.priority})"


@dataclass
class Conflict:
    """Represents a conflict between agents or resources"""
    conflict_id: str
    agent_ids: List[str]
    conflict_type: ConflictType
    description: str
    severity: int  # 0-10 scale
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Conflict(id={self.conflict_id}, type={self.conflict_type.value}, severity={self.severity})"


class ConflictResolver:
    """
    Resolves conflicts between agents using various resolution strategies.
    
    Attributes:
        resolution_strategies: Dictionary mapping conflict types to resolution functions
        resolution_history: History of resolved conflicts
    """

    def __init__(self):
        """Initialize the ConflictResolver with default strategies"""
        self.resolution_strategies: Dict[ConflictType, Callable] = {}
        self.resolution_history: List[Tuple[Conflict, str]] = []
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register default conflict resolution strategies"""
        self.resolution_strategies[ConflictType.RESOURCE_CONFLICT] = self._resolve_resource_conflict
        self.resolution_strategies[ConflictType.SCHEDULING_CONFLICT] = self._resolve_scheduling_conflict
        self.resolution_strategies[ConflictType.PRIORITY_CONFLICT] = self._resolve_priority_conflict
        self.resolution_strategies[ConflictType.GOAL_CONFLICT] = self._resolve_goal_conflict
        self.resolution_strategies[ConflictType.COMMUNICATION_CONFLICT] = self._resolve_communication_conflict

    def register_strategy(self, conflict_type: ConflictType, strategy: Callable) -> None:
        """
        Register a custom conflict resolution strategy.
        
        Args:
            conflict_type: Type of conflict to handle
            strategy: Callable that takes a Conflict and returns a resolution string
        """
        self.resolution_strategies[conflict_type] = strategy
        logger.info(f"Registered strategy for {conflict_type.value}")

    def resolve(self, conflict: Conflict) -> str:
        """
        Resolve a conflict using the appropriate strategy.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution description
        """
        if conflict.conflict_type not in self.resolution_strategies:
            resolution = self._resolve_generic_conflict(conflict)
        else:
            strategy = self.resolution_strategies[conflict.conflict_type]
            resolution = strategy(conflict)

        self.resolution_history.append((conflict, resolution))
        logger.info(f"Resolved conflict {conflict.conflict_id}: {resolution}")
        return resolution

    def _resolve_resource_conflict(self, conflict: Conflict) -> str:
        """Resolve resource conflicts by prioritizing agents"""
        priority_order = sorted(conflict.agent_ids)
        winner = priority_order[0]
        return f"Resource allocated to agent {winner} based on priority"

    def _resolve_scheduling_conflict(self, conflict: Conflict) -> str:
        """Resolve scheduling conflicts by reordering tasks"""
        return f"Scheduling conflict resolved by reordering tasks for agents {conflict.agent_ids}"

    def _resolve_priority_conflict(self, conflict: Conflict) -> str:
        """Resolve priority conflicts by escalating to higher authority"""
        return f"Priority conflict escalated for manual review among agents {conflict.agent_ids}"

    def _resolve_goal_conflict(self, conflict: Conflict) -> str:
        """Resolve goal conflicts by finding common ground"""
        return f"Goal conflict resolved by identifying common objectives for agents {conflict.agent_ids}"

    def _resolve_communication_conflict(self, conflict: Conflict) -> str:
        """Resolve communication conflicts by establishing clear protocols"""
        return f"Communication conflict resolved by establishing communication protocol between agents"

    def _resolve_generic_conflict(self, conflict: Conflict) -> str:
        """Generic conflict resolution fallback"""
        return f"Generic conflict resolution applied for conflict {conflict.conflict_id}"

    def get_resolution_history(self) -> List[Tuple[Conflict, str]]:
        """Get the history of resolved conflicts"""
        return self.resolution_history.copy()


class NegotiationAgent:
    """
    Represents an agent capable of negotiating with other agents.
    
    Attributes:
        agent_id: Unique identifier for the agent
        proposals: Dictionary of proposals sent by this agent
        negotiation_state: Current state of negotiation
    """

    def __init__(self, agent_id: str, preferences: Optional[Dict[str, Any]] = None):
        """
        Initialize a NegotiationAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            preferences: Optional dictionary of agent preferences
        """
        self.agent_id = agent_id
        self.preferences = preferences or {}
        self.proposals: Dict[str, Proposal] = {}
        self.received_proposals: Dict[str, Proposal] = {}
        self.negotiation_state = NegotiationState.INITIATING
        self.accepted_proposals: List[Proposal] = []
        self.rejected_proposals: List[Proposal] = []

    def create_proposal(self, proposal_id: str, content: Dict[str, Any],
                       priority: int = 0, metadata: Optional[Dict[str, Any]] = None) -> Proposal:
        """
        Create a negotiation proposal.
        
        Args:
            proposal_id: Unique identifier for the proposal
            content: Content of the proposal
            priority: Priority level of the proposal
            metadata: Optional metadata
            
        Returns:
            The created Proposal object
        """
        import time
        proposal = Proposal(
            proposer_id=self.agent_id,
            proposal_id=proposal_id,
            content=content,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata or {}
        )
        self.proposals[proposal_id] = proposal
        self.negotiation_state = NegotiationState.PROPOSING
        logger.info(f"Agent {self.agent_id} created proposal {proposal_id}")
        return proposal

    def evaluate_proposal(self, proposal: Proposal) -> bool:
        """
        Evaluate a received proposal against agent preferences.
        
        Args:
            proposal: Proposal to evaluate
            
        Returns:
            True if proposal is acceptable, False otherwise
        """
        # Simple evaluation: check if proposal aligns with preferences
        alignment_score = 0
        for key, preference_value in self.preferences.items():
            if key in proposal.content and proposal.content[key] == preference_value:
                alignment_score += 1

        acceptance_threshold = 0.5 * len(self.preferences) if self.preferences else 0
        return alignment_score >= acceptance_threshold

    def accept_proposal(self, proposal: Proposal) -> bool:
        """
        Accept a proposal.
        
        Args:
            proposal: Proposal to accept
            
        Returns:
            True if accepted, False if already processed
        """
        if proposal.proposal_id in self.accepted_proposals or proposal.proposal_id in self.rejected_proposals:
            return False

        self.accepted_proposals.append(proposal)
        self.negotiation_state = NegotiationState.ACCEPTING
        logger.info(f"Agent {self.agent_id} accepted proposal {proposal.proposal_id}")
        return True

    def reject_proposal(self, proposal: Proposal, reason: str = "") -> bool:
        """
        Reject a proposal.
        
        Args:
            proposal: Proposal to reject
            reason: Optional reason for rejection
            
        Returns:
            True if rejected, False if already processed
        """
        if proposal.proposal_id in self.accepted_proposals or proposal.proposal_id in self.rejected_proposals:
            return False

        self.rejected_proposals.append(proposal)
        self.negotiation_state = NegotiationState.REJECTING
        logger.info(f"Agent {self.agent_id} rejected proposal {proposal.proposal_id}: {reason}")
        return True

    def receive_proposal(self, proposal: Proposal) -> None:
        """
        Receive a proposal from another agent.
        
        Args:
            proposal: Proposal to receive
        """
        self.received_proposals[proposal.proposal_id] = proposal
        logger.info(f"Agent {self.agent_id} received proposal {proposal.proposal_id} from {proposal.proposer_id}")

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the negotiation agent"""
        return {
            "agent_id": self.agent_id,
            "state": self.negotiation_state.value,
            "proposals_sent": len(self.proposals),
            "proposals_accepted": len(self.accepted_proposals),
            "proposals_rejected": len(self.rejected_proposals),
            "proposals_received": len(self.received_proposals)
        }

    def __repr__(self) -> str:
        return f"NegotiationAgent(id={self.agent_id}, state={self.negotiation_state.value})"


class MultiAgentCoordinator:
    """
    Coordinates interactions between multiple agents.
    
    Manages agent registration, proposal routing, conflict resolution, and negotiation oversight.
    """

    def __init__(self):
        """Initialize the MultiAgentCoordinator"""
        self.agents: Dict[str, NegotiationAgent] = {}
        self.conflict_resolver = ConflictResolver()
        self.proposal_queue: List[Proposal] = []
        self.coordination_log: List[Dict[str, Any]] = []

    def register_agent(self, agent: NegotiationAgent) -> bool:
        """
        Register an agent with the coordinator.
        
        Args:
            agent: Agent to register
            
        Returns:
            True if successfully registered, False if agent ID already exists
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} is already registered")
            return False

        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id}")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the coordinator.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if successfully unregistered, False if agent not found
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False

        del self.agents[agent_id]
        logger.info(f"Unregistered agent {agent_id}")
        return True

    def submit_proposal(self, proposal: Proposal, target_agent_id: str) -> bool:
        """
        Submit a proposal from one agent to another.
        
        Args:
            proposal: Proposal to submit
            target_agent_id: ID of the target agent
            
        Returns:
            True if proposal delivered, False if target not found
        """
        if target_agent_id not in self.agents:
            logger.error(f"Target agent {target_agent_id} not found")
            return False

        target_agent = self.agents[target_agent_id]
        target_agent.receive_proposal(proposal)
        self.proposal_queue.append(proposal)
        self.coordination_log.append({
            "action": "proposal_submitted",
            "proposal_id": proposal.proposal_id,
            "from": proposal.proposer_id,
            "to": target_agent_id,
            "timestamp": proposal.timestamp
        })
        return True

    def detect_conflicts(self) -> List[Conflict]:
        """
        Detect potential conflicts between agents.
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        agent_ids = list(self.agents.keys())

        # Simple conflict detection: check for overlapping resource requests
        for i, agent1_id in enumerate(agent_ids):
            for agent2_id in agent_ids[i+1:]:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]

                # Check for conflicting proposals
                for prop1 in agent1.proposals.values():
                    for prop2 in agent2.proposals.values():
                        if self._proposals_conflict(prop1, prop2):
                            import time
                            conflict = Conflict(
                                conflict_id=f"{agent1_id}_{agent2_id}_{int(time.time() * 1000)}",
                                agent_ids=[agent1_id, agent2_id],
                                conflict_type=ConflictType.RESOURCE_CONFLICT,
                                description=f"Potential resource conflict between {agent1_id} and {agent2_id}",
                                severity=5,
                                timestamp=time.time()
                            )
                            conflicts.append(conflict)

        return conflicts

    def _proposals_conflict(self, prop1: Proposal, prop2: Proposal) -> bool:
        """
        Check if two proposals conflict with each other.
        
        Args:
            prop1: First proposal
            prop2: Second proposal
            
        Returns:
            True if proposals conflict, False otherwise
        """
        # Check for common resources in proposal content
        for key in prop1.content:
            if key in prop2.content and prop1.content[key] == prop2.content[key]:
                return True
        return False

    def resolve_conflict(self, conflict: Conflict) -> str:
        """
        Resolve a detected conflict.
        
        Args:
            conflict: Conflict to resolve
            
        Returns:
            Resolution description
        """
        resolution = self.conflict_resolver.resolve(conflict)
        self.coordination_log.append({
            "action": "conflict_resolved",
            "conflict_id": conflict.conflict_id,
            "resolution": resolution,
            "timestamp": conflict.timestamp
        })
        return resolution

    def coordinate_negotiation(self) -> Dict[str, Any]:
        """
        Coordinate ongoing negotiations between agents.
        
        Returns:
            Dictionary with negotiation status and summary
        """
        agent_statuses = {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
        conflicts = self.detect_conflicts()

        for conflict in conflicts:
            self.resolve_conflict(conflict)

        return {
            "total_agents": len(self.agents),
            "agent_statuses": agent_statuses,
            "detected_conflicts": len(conflicts),
            "resolved_conflicts": len(self.conflict_resolver.get_resolution_history()),
            "proposals_in_queue": len(self.proposal_queue)
        }

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get the overall coordination status"""
        return {
            "registered_agents": len(self.agents),
            "agent_ids": list(self.agents.keys()),
            "proposal_queue_size": len(self.proposal_queue),
            "coordination_log_entries": len(self.coordination_log),
            "total_conflicts_resolved": len(self.conflict_resolver.get_resolution_history())
        }

    def __repr__(self) -> str:
        return f"MultiAgentCoordinator(agents={len(self.agents)}, proposals={len(self.proposal_queue)})"


# Export public API
__all__ = [
    "MultiAgentCoordinator",
    "NegotiationAgent",
    "ConflictResolver",
    "Proposal",
    "Conflict",
    "NegotiationState",
    "ConflictType"
]
