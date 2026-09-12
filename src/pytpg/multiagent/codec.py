"""Reversible mixed-radix encoding for centralized joint actions."""

from __future__ import annotations

from collections.abc import Mapping

from pytpg.multiagent._validation import validate_specs
from pytpg.multiagent.errors import MultiAgentActionError
from pytpg.multiagent.model import AgentAction, AgentID, AgentSpec


class JointActionCodec:
    """Map ordered heterogeneous scalar actions to one atomic action ID."""

    __slots__ = ("_action_count", "specs")

    def __init__(self, specs: tuple[AgentSpec, ...]) -> None:
        self.specs = validate_specs(specs)
        action_count = 1
        for spec in specs:
            action_count *= spec.action_count
        self._action_count = action_count

    @property
    def action_count(self) -> int:
        """Return the number of representable joint actions."""

        return self._action_count

    def encode(self, actions: Mapping[AgentID, int]) -> int:
        """Encode one action per configured agent in declared agent order."""

        if not isinstance(actions, Mapping):
            raise MultiAgentActionError("joint actions must be a mapping by agent ID")
        expected = {spec.agent_id for spec in self.specs}
        actual = set(actions)
        if actual != expected:
            raise MultiAgentActionError(
                "joint actions must contain exactly the configured agent IDs"
            )

        encoded = 0
        for spec in self.specs:
            action = actions[spec.agent_id]
            self._validate_agent_action(spec, action)
            encoded = encoded * spec.action_count + action
        return encoded

    def decode(self, joint_action_id: int) -> tuple[AgentAction, ...]:
        """Decode one atomic ID into actions in declared agent order."""

        if (
            isinstance(joint_action_id, bool)
            or not isinstance(joint_action_id, int)
            or not 0 <= joint_action_id < self.action_count
        ):
            raise MultiAgentActionError(
                f"joint action {joint_action_id!r} is outside [0, {self.action_count})"
            )

        remaining = joint_action_id
        reversed_actions: list[AgentAction] = []
        for spec in reversed(self.specs):
            action_id = remaining % spec.action_count
            remaining //= spec.action_count
            reversed_actions.append(AgentAction(spec.agent_id, action_id))
        return tuple(reversed(reversed_actions))

    @staticmethod
    def _validate_agent_action(spec: AgentSpec, action: object) -> None:
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < spec.action_count
        ):
            raise MultiAgentActionError(
                f"action for agent {spec.agent_id!r} is outside "
                f"[0, {spec.action_count})"
            )


__all__ = ["JointActionCodec"]
