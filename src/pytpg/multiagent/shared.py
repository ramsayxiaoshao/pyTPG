"""Centralized shared-control TPG over an ordered fixed agent roster."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pytpg.core import TPGGraph
from pytpg.memory import (
    Memory,
    MemorySnapshot,
    NullMemory,
    StatefulGraphRuntime,
    StatefulTraversalResult,
)
from pytpg.multiagent._validation import ordered_observations, validate_specs
from pytpg.multiagent.codec import JointActionCodec
from pytpg.multiagent.controller import (
    validate_controller_root,
    validate_graph_actions,
)
from pytpg.multiagent.errors import MultiAgentConfigurationError, MultiAgentStateError
from pytpg.multiagent.model import AgentAction, AgentID, AgentSpec, action_mapping
from pytpg.runtime import GraphRuntime, OperatorRegistry


@dataclass(frozen=True, slots=True)
class SharedControlStep:
    """One centralized traversal and its decoded per-agent actions."""

    joint_action_id: int
    actions: tuple[AgentAction, ...]
    traversal: StatefulTraversalResult

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple) or not all(
            isinstance(item, AgentAction) for item in self.actions
        ):
            raise MultiAgentStateError("shared-control actions must be a tuple")
        if not isinstance(self.traversal, StatefulTraversalResult):
            raise MultiAgentStateError("shared-control traversal is invalid")
        if self.joint_action_id != self.traversal.action_id:
            raise MultiAgentStateError(
                "joint action does not match its shared-control traversal"
            )
        if not self.actions:
            raise MultiAgentStateError("a shared-control step cannot be empty")
        agent_ids = tuple(action.agent_id for action in self.actions)
        if len(set(agent_ids)) != len(agent_ids):
            raise MultiAgentStateError("shared-control step agent IDs must be unique")

    @property
    def actions_by_agent(self) -> dict[AgentID, int]:
        """Return a fresh decoded action mapping in configured agent order."""

        return action_mapping(self.actions)


class SharedControlRuntime:
    """Use one graph and one memory state to choose an encoded joint action."""

    __slots__ = (
        "_episode_active",
        "codec",
        "graph",
        "root_team_id",
        "specs",
        "stateful_runtime",
    )

    def __init__(
        self,
        specs: tuple[AgentSpec, ...],
        graph: TPGGraph,
        stateful_runtime: StatefulGraphRuntime,
        *,
        root_team_id: int | None = None,
    ) -> None:
        selected_specs = validate_specs(specs)
        if not isinstance(graph, TPGGraph):
            raise MultiAgentConfigurationError("graph must be a TPGGraph")
        if not isinstance(stateful_runtime, StatefulGraphRuntime):
            raise MultiAgentConfigurationError(
                "stateful_runtime must be a StatefulGraphRuntime"
            )
        joint_observation_size = sum(spec.observation_size for spec in selected_specs)
        if stateful_runtime.observation_size != joint_observation_size:
            raise MultiAgentConfigurationError(
                "shared runtime observation_size must equal the sum of agent widths"
            )
        if stateful_runtime.episode_active:
            raise MultiAgentConfigurationError(
                "a shared controller must be constructed with an inactive runtime"
            )

        codec = JointActionCodec(selected_specs)
        validate_controller_root(graph, root_team_id)
        stateful_runtime.runtime.validate(graph).require_valid()
        validate_graph_actions(graph, codec.action_count)

        self.specs = selected_specs
        self.graph = graph
        self.stateful_runtime = stateful_runtime
        self.root_team_id = root_team_id
        self.codec = codec
        self._episode_active = False

    @classmethod
    def create(
        cls,
        specs: tuple[AgentSpec, ...],
        graph: TPGGraph,
        *,
        memory: Memory | None = None,
        operators: OperatorRegistry | None = None,
        max_steps: int | None = None,
        root_team_id: int | None = None,
    ) -> SharedControlRuntime:
        """Infer one runtime over concatenated observation and memory inputs."""

        selected_specs = validate_specs(specs)
        if not isinstance(graph, TPGGraph):
            raise MultiAgentConfigurationError("graph must be a TPGGraph")
        selected_memory = NullMemory() if memory is None else memory
        observation_size = sum(spec.observation_size for spec in selected_specs)
        runtime = GraphRuntime.infer_for_graph(
            graph,
            (0.0,) * (observation_size + selected_memory.size),
            operators,
            max_steps=max_steps,
        )
        return cls(
            selected_specs,
            graph,
            StatefulGraphRuntime(runtime, selected_memory, observation_size),
            root_team_id=root_team_id,
        )

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    @property
    def memory(self) -> Memory:
        return self.stateful_runtime.memory

    def reset_episode(self) -> MemorySnapshot:
        if self._episode_active:
            raise MultiAgentStateError("a shared-control episode is already active")
        snapshot = self.stateful_runtime.reset_episode()
        self._episode_active = True
        return snapshot

    def step(
        self, observations: Mapping[AgentID, Sequence[float]]
    ) -> SharedControlStep:
        """Concatenate observations, traverse once, and decode the joint action."""

        self._require_active()
        ordered = ordered_observations(self.specs, observations)
        joint_observation = tuple(value for item in ordered for value in item)
        traversal = self.stateful_runtime.traverse(
            self.graph,
            joint_observation,
            root_team_id=self.root_team_id,
        )
        actions = self.codec.decode(traversal.action_id)
        return SharedControlStep(traversal.action_id, actions, traversal)

    def end_episode(self) -> MemorySnapshot:
        self._require_active()
        before = self.memory.snapshot()
        try:
            snapshot = self.stateful_runtime.end_episode()
        except Exception as error:
            try:
                self.stateful_runtime.restore_episode(before)
            except Exception:
                raise MultiAgentStateError(
                    "episode end failed and shared memory could not resume"
                ) from error
            raise
        self._episode_active = False
        return snapshot

    def _require_active(self) -> None:
        if not self._episode_active:
            raise MultiAgentStateError(
                "reset_episode is required before this operation"
            )
        if not self.stateful_runtime.episode_active:
            raise MultiAgentStateError(
                "the bound controller was modified outside its shared-control runtime"
            )


__all__ = ["SharedControlRuntime", "SharedControlStep"]
