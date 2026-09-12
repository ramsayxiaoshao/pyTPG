"""Immutable aggregate for a structurally valid Tangled Program Graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pytpg.core._validation import require_non_negative_integer
from pytpg.core.action import TeamReference
from pytpg.core.identifiers import ActionID, LearnerID, ProgramID, TeamID
from pytpg.core.learner import Learner
from pytpg.core.program import Program
from pytpg.core.team import Team

if TYPE_CHECKING:
    from pytpg.runtime import (
        GraphRuntime,
        GraphSummary,
        GraphValidationReport,
        OperatorRegistry,
        RuntimeConfig,
        TraversalResult,
    )


@dataclass(frozen=True, slots=True)
class TPGGraph:
    """Teams and explicit roots with context-free invariants enforced.

    Operator existence, operand bounds, reachability, and traversal termination
    require future runtime policy and are intentionally not checked here.
    """

    teams: tuple[Team, ...]
    root_team_ids: tuple[TeamID, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.teams, tuple):
            raise TypeError("graph teams must be a tuple")
        if not isinstance(self.root_team_ids, tuple):
            raise TypeError("graph root IDs must be a tuple")
        if not self.teams:
            raise ValueError("graph must contain at least one team")
        if not self.root_team_ids:
            raise ValueError("graph must contain at least one root team ID")
        if not all(isinstance(item, Team) for item in self.teams):
            raise TypeError("graph teams contain an unsupported value")

        teams_by_id = self._index_teams()
        self._check_roots(teams_by_id)
        self._check_references(teams_by_id)
        self._check_reused_entities()

    def _index_teams(self) -> dict[TeamID, Team]:
        teams_by_id: dict[TeamID, Team] = {}
        for team in self.teams:
            if team.id in teams_by_id:
                msg = f"graph repeats team ID {team.id}"
                raise ValueError(msg)
            teams_by_id[team.id] = team
        return teams_by_id

    def _check_roots(self, teams_by_id: dict[TeamID, Team]) -> None:
        root_ids: set[TeamID] = set()
        for root_id in self.root_team_ids:
            require_non_negative_integer(root_id, "root team ID")
            if root_id in root_ids:
                msg = f"graph repeats root team ID {root_id}"
                raise ValueError(msg)
            if root_id not in teams_by_id:
                msg = f"root team ID {root_id} does not resolve"
                raise ValueError(msg)
            root_ids.add(root_id)

    def _check_references(self, teams_by_id: dict[TeamID, Team]) -> None:
        for team in self.teams:
            for learner in team.learners:
                if (
                    isinstance(learner.action, TeamReference)
                    and learner.action.team_id not in teams_by_id
                ):
                    msg = (
                        f"learner {learner.id} references missing team "
                        f"{learner.action.team_id}"
                    )
                    raise ValueError(msg)

    def _check_reused_entities(self) -> None:
        learners_by_id: dict[LearnerID, Learner] = {}
        programs_by_id: dict[ProgramID, Program] = {}

        for team in self.teams:
            for learner in team.learners:
                existing_learner = learners_by_id.get(learner.id)
                if existing_learner is not None and existing_learner != learner:
                    msg = f"learner ID {learner.id} denotes unequal values"
                    raise ValueError(msg)
                learners_by_id[learner.id] = learner

                program = learner.program
                existing_program = programs_by_id.get(program.id)
                if existing_program is not None and existing_program != program:
                    msg = f"program ID {program.id} denotes unequal values"
                    raise ValueError(msg)
                programs_by_id[program.id] = program

    def validate(
        self,
        config: RuntimeConfig | None = None,
        *,
        operators: OperatorRegistry | None = None,
    ) -> GraphValidationReport:
        """Collect graph diagnostics, inferring dimensions when omitted."""

        from pytpg.runtime.graph_validation import GraphValidator

        return GraphValidator(config, operators).validate(self)

    def summary(self) -> GraphSummary:
        """Return deterministic ID-deduplicated structural statistics."""

        from pytpg.runtime.inspection import summarize_graph

        return summarize_graph(self)

    def traverse(
        self,
        observation: Sequence[float],
        *,
        root_team_id: TeamID | None = None,
        runtime: GraphRuntime | None = None,
    ) -> TraversalResult:
        """Execute graph traversal and retain an inspectable decision trace."""

        if runtime is None:
            from pytpg.runtime.graph_runtime import GraphRuntime

            runtime = GraphRuntime.infer_for_graph(self, observation)
        return runtime.traverse(self, observation, root_team_id=root_team_id)

    def act(
        self,
        observation: Sequence[float],
        *,
        root_team_id: TeamID | None = None,
        runtime: GraphRuntime | None = None,
    ) -> ActionID:
        """Return the terminal atomic action selected by graph traversal."""

        return ActionID(
            self.traverse(
                observation,
                root_team_id=root_team_id,
                runtime=runtime,
            ).action_id
        )


__all__ = ["TPGGraph"]
