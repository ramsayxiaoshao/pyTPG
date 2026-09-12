"""Transactional synchronous execution of per-agent TPG controllers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pytpg.memory import MemorySnapshot, StatefulTraversalResult
from pytpg.multiagent._validation import ordered_observations
from pytpg.multiagent.controller import TPGAgentController
from pytpg.multiagent.errors import (
    MultiAgentConfigurationError,
    MultiAgentStateError,
)
from pytpg.multiagent.model import AgentAction, AgentID, action_mapping


@dataclass(frozen=True, slots=True)
class IndependentStep:
    """Aligned actions and per-agent traversals for one synchronous step."""

    actions: tuple[AgentAction, ...]
    traversals: tuple[StatefulTraversalResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple) or not all(
            isinstance(item, AgentAction) for item in self.actions
        ):
            raise MultiAgentStateError("independent actions must be a tuple")
        if not isinstance(self.traversals, tuple) or not all(
            isinstance(item, StatefulTraversalResult) for item in self.traversals
        ):
            raise MultiAgentStateError("independent traversals must be a tuple")
        if len(self.actions) != len(self.traversals):
            raise MultiAgentStateError(
                "independent actions and traversals must have equal lengths"
            )
        if not self.actions:
            raise MultiAgentStateError("an independent step cannot be empty")
        agent_ids = tuple(action.agent_id for action in self.actions)
        if len(set(agent_ids)) != len(agent_ids):
            raise MultiAgentStateError("independent step agent IDs must be unique")
        for action, traversal in zip(self.actions, self.traversals, strict=True):
            if action.action_id != traversal.action_id:
                raise MultiAgentStateError(
                    "an independent action does not match its traversal"
                )

    @property
    def actions_by_agent(self) -> dict[AgentID, int]:
        """Return a fresh action mapping in configured agent order."""

        return action_mapping(self.actions)


class IndependentMultiAgentRuntime:
    """Run fixed per-agent controllers with atomic joint-step memory updates."""

    __slots__ = ("_episode_active", "controllers", "specs")

    def __init__(self, controllers: tuple[TPGAgentController, ...]) -> None:
        if not isinstance(controllers, tuple):
            raise MultiAgentConfigurationError("controllers must be a tuple")
        if not controllers:
            raise MultiAgentConfigurationError("at least one controller is required")
        if not all(isinstance(item, TPGAgentController) for item in controllers):
            raise MultiAgentConfigurationError(
                "controllers contain an unsupported value"
            )

        agent_ids = tuple(controller.spec.agent_id for controller in controllers)
        if len(set(agent_ids)) != len(agent_ids):
            raise MultiAgentConfigurationError("controller agent IDs must be unique")
        stateful_ids = tuple(id(item.stateful_runtime) for item in controllers)
        if len(set(stateful_ids)) != len(stateful_ids):
            raise MultiAgentConfigurationError(
                "agents must not share a stateful runtime or live memory owner"
            )
        memory_ids = tuple(id(item.memory) for item in controllers)
        if len(set(memory_ids)) != len(memory_ids):
            raise MultiAgentConfigurationError(
                "agents must not share a stateful runtime or live memory owner"
            )
        if any(controller.episode_active for controller in controllers):
            raise MultiAgentConfigurationError(
                "controllers must be inactive when assigned to a multi-agent runtime"
            )

        self.controllers = controllers
        self.specs = tuple(controller.spec for controller in controllers)
        self._episode_active = False

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def reset_episode(self) -> tuple[MemorySnapshot, ...]:
        """Reset every controller in deterministic roster order."""

        if self._episode_active:
            raise MultiAgentStateError("a multi-agent episode is already active")
        reset_controllers: list[TPGAgentController] = []
        snapshots: list[MemorySnapshot] = []
        try:
            for controller in self.controllers:
                snapshots.append(controller.reset_episode())
                reset_controllers.append(controller)
        except Exception as error:
            cleanup_failures: list[Exception] = []
            for controller in reversed(reset_controllers):
                try:
                    controller.end_episode()
                except Exception as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            if cleanup_failures:
                raise MultiAgentStateError(
                    "episode reset failed and partial resets could not be ended"
                ) from error
            raise
        self._episode_active = True
        return tuple(snapshots)

    def step(self, observations: Mapping[AgentID, Sequence[float]]) -> IndependentStep:
        """Execute one all-agent step and roll every memory back on failure."""

        self._require_active()
        normalized = ordered_observations(self.specs, observations)
        snapshots = tuple(
            controller.memory.snapshot() for controller in self.controllers
        )
        traversals: list[StatefulTraversalResult] = []
        try:
            for controller, observation in zip(
                self.controllers, normalized, strict=True
            ):
                traversals.append(controller.traverse(observation))
        except Exception as error:
            rollback_failures: list[Exception] = []
            for controller, snapshot in zip(self.controllers, snapshots, strict=True):
                try:
                    controller.restore_episode(snapshot)
                except Exception as rollback_error:
                    rollback_failures.append(rollback_error)
            if rollback_failures:
                raise MultiAgentStateError(
                    "joint step failed and one or more memories could not be restored"
                ) from error
            raise

        actions = tuple(
            AgentAction(controller.spec.agent_id, traversal.action_id)
            for controller, traversal in zip(self.controllers, traversals, strict=True)
        )
        return IndependentStep(actions, tuple(traversals))

    def end_episode(self) -> tuple[MemorySnapshot, ...]:
        """End every controller episode in deterministic roster order."""

        self._require_active()
        before = tuple(controller.memory.snapshot() for controller in self.controllers)
        snapshots: list[MemorySnapshot] = []
        try:
            for controller in self.controllers:
                snapshots.append(controller.end_episode())
        except Exception as error:
            rollback_failures: list[Exception] = []
            for controller, snapshot in zip(self.controllers, before, strict=True):
                try:
                    controller.restore_episode(snapshot)
                except Exception as rollback_error:
                    rollback_failures.append(rollback_error)
            if rollback_failures:
                raise MultiAgentStateError(
                    "episode end failed and one or more controllers could not resume"
                ) from error
            raise
        self._episode_active = False
        return tuple(snapshots)

    def _require_active(self) -> None:
        if not self._episode_active:
            raise MultiAgentStateError(
                "reset_episode is required before this operation"
            )
        if not all(controller.episode_active for controller in self.controllers):
            raise MultiAgentStateError(
                "a bound controller was modified outside its multi-agent runtime"
            )


__all__ = ["IndependentMultiAgentRuntime", "IndependentStep"]
