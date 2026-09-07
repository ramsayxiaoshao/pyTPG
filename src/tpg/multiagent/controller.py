"""A graph and stateful runtime bound to one declared agent."""

from __future__ import annotations

from tpg.core import AtomicAction, TPGGraph
from tpg.memory import (
    Memory,
    MemorySnapshot,
    NullMemory,
    StatefulGraphRuntime,
    StatefulTraversalResult,
)
from tpg.multiagent.errors import (
    MultiAgentActionError,
    MultiAgentConfigurationError,
)
from tpg.multiagent.model import AgentSpec
from tpg.runtime import GraphRuntime, OperatorRegistry


class TPGAgentController:
    """Own one agent's graph binding and independent episode memory."""

    __slots__ = ("graph", "root_team_id", "spec", "stateful_runtime")

    def __init__(
        self,
        spec: AgentSpec,
        graph: TPGGraph,
        stateful_runtime: StatefulGraphRuntime,
        *,
        root_team_id: int | None = None,
    ) -> None:
        if not isinstance(spec, AgentSpec):
            raise MultiAgentConfigurationError("spec must be an AgentSpec")
        if not isinstance(graph, TPGGraph):
            raise MultiAgentConfigurationError("graph must be a TPGGraph")
        if not isinstance(stateful_runtime, StatefulGraphRuntime):
            raise MultiAgentConfigurationError(
                "stateful_runtime must be a StatefulGraphRuntime"
            )
        if stateful_runtime.episode_active:
            raise MultiAgentConfigurationError(
                "a controller must be constructed with an inactive runtime"
            )
        if stateful_runtime.observation_size != spec.observation_size:
            raise MultiAgentConfigurationError(
                f"agent {spec.agent_id!r} observation_size does not match its runtime"
            )
        validate_controller_root(graph, root_team_id)
        stateful_runtime.runtime.validate(graph).require_valid()
        validate_graph_actions(graph, spec.action_count)
        self.spec = spec
        self.graph = graph
        self.stateful_runtime = stateful_runtime
        self.root_team_id = root_team_id

    @classmethod
    def create(
        cls,
        spec: AgentSpec,
        graph: TPGGraph,
        *,
        memory: Memory | None = None,
        operators: OperatorRegistry | None = None,
        max_steps: int | None = None,
        root_team_id: int | None = None,
    ) -> TPGAgentController:
        """Infer the runtime shape and bind optional episode memory."""

        if not isinstance(spec, AgentSpec):
            raise MultiAgentConfigurationError("spec must be an AgentSpec")
        if not isinstance(graph, TPGGraph):
            raise MultiAgentConfigurationError("graph must be a TPGGraph")
        selected_memory = NullMemory() if memory is None else memory
        input_size = spec.observation_size + selected_memory.size
        runtime = GraphRuntime.infer_for_graph(
            graph,
            (0.0,) * input_size,
            operators,
            max_steps=max_steps,
        )
        return cls(
            spec,
            graph,
            StatefulGraphRuntime(runtime, selected_memory, spec.observation_size),
            root_team_id=root_team_id,
        )

    @property
    def episode_active(self) -> bool:
        return self.stateful_runtime.episode_active

    @property
    def memory(self) -> Memory:
        return self.stateful_runtime.memory

    def reset_episode(self) -> MemorySnapshot:
        return self.stateful_runtime.reset_episode()

    def end_episode(self) -> MemorySnapshot:
        return self.stateful_runtime.end_episode()

    def restore_episode(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        return self.stateful_runtime.restore_episode(snapshot)

    def traverse(self, observation: tuple[float, ...]) -> StatefulTraversalResult:
        result = self.stateful_runtime.traverse(
            self.graph,
            observation,
            root_team_id=self.root_team_id,
        )
        if not 0 <= result.action_id < self.spec.action_count:
            raise MultiAgentActionError(
                f"graph returned action {result.action_id} outside agent "
                f"{self.spec.agent_id!r} space [0, {self.spec.action_count})"
            )
        return result


def validate_controller_root(graph: TPGGraph, root_team_id: int | None) -> None:
    """Validate a controller's fixed root selection."""

    if root_team_id is None:
        if len(graph.root_team_ids) != 1:
            raise MultiAgentConfigurationError(
                "root_team_id is required for a multi-root controller"
            )
        return
    if (
        isinstance(root_team_id, bool)
        or not isinstance(root_team_id, int)
        or root_team_id not in graph.root_team_ids
    ):
        raise MultiAgentConfigurationError(
            f"root_team_id {root_team_id!r} is not a declared graph root"
        )


def validate_graph_actions(graph: TPGGraph, action_count: int) -> None:
    """Require every graph atomic action to fit the declared action space."""

    for team in graph.teams:
        for learner in team.learners:
            if (
                isinstance(learner.action, AtomicAction)
                and learner.action.action_id >= action_count
            ):
                raise MultiAgentConfigurationError(
                    f"graph atomic action {learner.action.action_id} is outside "
                    f"[0, {action_count})"
                )


__all__ = [
    "TPGAgentController",
    "validate_controller_root",
    "validate_graph_actions",
]
