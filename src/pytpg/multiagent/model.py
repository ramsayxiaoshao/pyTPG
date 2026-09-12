"""Immutable values shared by independent and centralized TPG control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from pytpg.multiagent.errors import MultiAgentActionError, MultiAgentConfigurationError

AgentID = NewType("AgentID", str)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """One agent's stable ID and fixed observation/action dimensions."""

    agent_id: AgentID
    observation_size: int
    action_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.agent_id, str)
            or not self.agent_id
            or self.agent_id.strip() != self.agent_id
        ):
            raise MultiAgentConfigurationError(
                "agent_id must be a non-empty string without surrounding whitespace"
            )
        if (
            isinstance(self.observation_size, bool)
            or not isinstance(self.observation_size, int)
            or self.observation_size < 0
        ):
            raise MultiAgentConfigurationError(
                "observation_size must be a non-negative integer"
            )
        if (
            isinstance(self.action_count, bool)
            or not isinstance(self.action_count, int)
            or self.action_count < 1
        ):
            raise MultiAgentConfigurationError(
                "action_count must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class AgentAction:
    """One decoded scalar action associated with its agent."""

    agent_id: AgentID
    action_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise MultiAgentActionError("agent action has an invalid agent ID")
        if (
            isinstance(self.action_id, bool)
            or not isinstance(self.action_id, int)
            or self.action_id < 0
        ):
            raise MultiAgentActionError(
                "agent action_id must be a non-negative integer"
            )


def action_mapping(actions: tuple[AgentAction, ...]) -> dict[AgentID, int]:
    """Return a fresh mapping while preserving tuple order on iteration."""

    return {action.agent_id: action.action_id for action in actions}


__all__ = ["AgentAction", "AgentID", "AgentSpec"]
