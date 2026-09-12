"""Safe deterministic traversal of a validated Tangled Program Graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pytpg.runtime._model import GraphLike, LearnerLike
from pytpg.runtime.config import RuntimeConfig
from pytpg.runtime.errors import (
    MissingTeamError,
    NoEligibleLearnerError,
    RootSelectionError,
    RuntimeConfigurationError,
    TraversalLimitExceededError,
)
from pytpg.runtime.graph_validation import GraphValidationReport, GraphValidator
from pytpg.runtime.inference import infer_register_count
from pytpg.runtime.inspection import GraphSummary, summarize_graph
from pytpg.runtime.operators import OperatorRegistry
from pytpg.runtime.team_runtime import DeterministicRuntime

TraversalActionKind: TypeAlias = Literal["atomic", "team_reference"]


@dataclass(frozen=True, slots=True)
class TraversalStep:
    """One winning learner decision in an inspectable traversal trace."""

    team_id: int
    learner_id: int
    bid: float
    atomic_action_id: int | None = None
    referenced_team_id: int | None = None

    def __post_init__(self) -> None:
        if (self.atomic_action_id is None) == (self.referenced_team_id is None):
            raise ValueError(
                "a traversal step must contain exactly one action destination"
            )

    @property
    def action_kind(self) -> TraversalActionKind:
        """Return the selected action variant without conflating its ID type."""

        return "atomic" if self.atomic_action_id is not None else "team_reference"


@dataclass(frozen=True, slots=True)
class TraversalResult:
    """The terminal atomic action and every decision that produced it."""

    action_id: int
    steps: tuple[TraversalStep, ...]

    @property
    def visited_team_ids(self) -> tuple[int, ...]:
        """Return visited teams in traversal order."""

        return tuple(step.team_id for step in self.steps)


class GraphRuntime:
    """Traverse team references with visited-edge exclusion and a hard limit."""

    __slots__ = ("max_steps", "runtime", "validate_before_execution")

    def __init__(
        self,
        config: RuntimeConfig,
        operators: OperatorRegistry | None = None,
        *,
        max_steps: int | None = None,
        validate_before_execution: bool = True,
    ) -> None:
        if max_steps is not None and (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 1
        ):
            msg = f"max_steps must be a positive integer or None, got {max_steps!r}"
            raise RuntimeConfigurationError(msg)
        self.runtime = DeterministicRuntime(config, operators)
        self.max_steps = max_steps
        self.validate_before_execution = validate_before_execution

    @property
    def config(self) -> RuntimeConfig:
        """Expose the immutable program-execution shape configuration."""

        return self.runtime.executor.config

    @classmethod
    def infer_for_graph(
        cls,
        graph: GraphLike,
        observation: Sequence[float],
        operators: OperatorRegistry | None = None,
        *,
        max_steps: int | None = None,
        validate_before_execution: bool = True,
    ) -> GraphRuntime:
        """Infer only dimensions needed by the graph convenience API."""

        return cls(
            RuntimeConfig(
                input_size=len(observation),
                register_count=infer_register_count(graph.teams),
            ),
            operators,
            max_steps=max_steps,
            validate_before_execution=validate_before_execution,
        )

    def validate(self, graph: GraphLike) -> GraphValidationReport:
        """Validate with this runtime's exact configuration and operators."""

        validator = GraphValidator(
            self.runtime.executor.config, self.runtime.executor.operators
        )
        return validator.validate(graph)

    @staticmethod
    def summary(graph: GraphLike) -> GraphSummary:
        """Return deterministic structural graph statistics."""

        return summarize_graph(graph)

    def traverse(
        self,
        graph: GraphLike,
        observation: Sequence[float],
        *,
        root_team_id: int | None = None,
    ) -> TraversalResult:
        """Traverse until an atomic action wins or a safety error is raised."""

        if self.validate_before_execution:
            self.validate(graph).require_valid()
        self.runtime.executor.normalize_observation(observation)

        teams_by_id = {team.id: team for team in graph.teams}
        current_team_id = self._resolve_root(graph, root_team_id)
        limit = self.max_steps or len(teams_by_id)
        visited: set[int] = set()
        steps: list[TraversalStep] = []

        for _ in range(limit):
            team = teams_by_id.get(current_team_id)
            if team is None:
                msg = f"team ID {current_team_id} does not resolve during traversal"
                raise MissingTeamError(msg)
            visited.add(current_team_id)
            eligible = tuple(
                learner
                for learner in team.learners
                if learner.action.kind == "atomic"
                or learner.action.team_id not in visited
            )
            if not eligible:
                msg = (
                    f"team {current_team_id} has no eligible learner after excluding "
                    "references to visited teams"
                )
                raise NoEligibleLearnerError(msg)

            winner, bid = self._select_eligible(eligible, observation)
            if winner.action.kind == "atomic":
                step = TraversalStep(
                    team_id=current_team_id,
                    learner_id=winner.id,
                    bid=bid,
                    atomic_action_id=winner.action.action_id,
                )
                steps.append(step)
                return TraversalResult(winner.action.action_id, tuple(steps))

            target = winner.action.team_id
            steps.append(
                TraversalStep(
                    team_id=current_team_id,
                    learner_id=winner.id,
                    bid=bid,
                    referenced_team_id=target,
                )
            )
            if target not in teams_by_id:
                msg = f"learner {winner.id} references missing team {target}"
                raise MissingTeamError(msg)
            current_team_id = target

        msg = f"graph traversal exceeded its {limit}-step limit"
        raise TraversalLimitExceededError(msg)

    def act(
        self,
        graph: GraphLike,
        observation: Sequence[float],
        *,
        root_team_id: int | None = None,
    ) -> int:
        """Return only the terminal atomic action ID."""

        return self.traverse(
            graph,
            observation,
            root_team_id=root_team_id,
        ).action_id

    def _select_eligible(
        self,
        learners: tuple[LearnerLike, ...],
        observation: Sequence[float],
    ) -> tuple[LearnerLike, float]:
        bids = tuple(
            self.runtime.executor.execute(learner.program, observation).output
            for learner in learners
        )
        winner_index = max(range(len(learners)), key=bids.__getitem__)
        return learners[winner_index], bids[winner_index]

    @staticmethod
    def _resolve_root(graph: GraphLike, root_team_id: int | None) -> int:
        if root_team_id is None:
            if len(graph.root_team_ids) != 1:
                msg = (
                    "root_team_id is required when a graph declares "
                    f"{len(graph.root_team_ids)} roots"
                )
                raise RootSelectionError(msg)
            return graph.root_team_ids[0]
        if root_team_id not in graph.root_team_ids:
            msg = f"team {root_team_id} is not a declared graph root"
            raise RootSelectionError(msg)
        return root_team_id


__all__ = [
    "GraphRuntime",
    "TraversalActionKind",
    "TraversalResult",
    "TraversalStep",
]
