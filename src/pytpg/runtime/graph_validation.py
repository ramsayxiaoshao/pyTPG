"""Collected graph validation for runtime and research diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pytpg.runtime._model import GraphLike, LearnerLike, ProgramLike, TeamLike
from pytpg.runtime.config import RuntimeConfig
from pytpg.runtime.errors import (
    InvalidInstructionError,
    OperatorArityError,
    TPGExecutionError,
    UnknownOperatorError,
)
from pytpg.runtime.executor import ProgramExecutor
from pytpg.runtime.inference import infer_validation_config
from pytpg.runtime.operators import OperatorRegistry

ValidationSeverity: TypeAlias = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class GraphValidationIssue:
    """One stable, machine-inspectable graph diagnostic."""

    severity: ValidationSeverity
    code: str
    message: str
    team_id: int | None = None
    learner_id: int | None = None
    program_id: int | None = None
    instruction_index: int | None = None


@dataclass(frozen=True, slots=True)
class GraphValidationReport:
    """All diagnostics produced by one deterministic validation pass."""

    issues: tuple[GraphValidationIssue, ...]

    @property
    def errors(self) -> tuple[GraphValidationIssue, ...]:
        """Return only validity-breaking diagnostics."""

        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[GraphValidationIssue, ...]:
        """Return non-breaking research and maintenance diagnostics."""

        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Return whether validation found no errors."""

        return not self.errors

    def require_valid(self) -> None:
        """Raise one exception containing this report when errors exist."""

        if not self.is_valid:
            raise GraphValidationError(self)


class GraphValidationError(TPGExecutionError):
    """A graph cannot execute because validation found one or more errors."""

    def __init__(self, report: GraphValidationReport) -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.errors)
        super().__init__(f"graph validation failed: {codes}")


class GraphValidator:
    """Validate graph structure and every program under one runtime context."""

    __slots__ = ("config", "operators")

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        operators: OperatorRegistry | None = None,
    ) -> None:
        self.config = config
        self.operators = operators

    def validate(self, graph: GraphLike) -> GraphValidationReport:
        """Collect deterministic errors and warnings without mutating the graph."""

        issues: list[GraphValidationIssue] = []
        teams_by_id = self._index_teams(graph, issues)
        self._validate_roots(graph, teams_by_id, issues)

        config = self.config or infer_validation_config(graph.teams)
        executor = ProgramExecutor(config, self.operators)
        learners_by_id: dict[int, LearnerLike] = {}
        programs_by_id: dict[int, ProgramLike] = {}

        for team in graph.teams:
            self._validate_team(
                team,
                teams_by_id,
                executor,
                learners_by_id,
                programs_by_id,
                issues,
            )

        self._report_orphans(graph, teams_by_id, issues)
        return GraphValidationReport(tuple(issues))

    @staticmethod
    def _index_teams(
        graph: GraphLike,
        issues: list[GraphValidationIssue],
    ) -> dict[int, TeamLike]:
        teams_by_id: dict[int, TeamLike] = {}
        if not graph.teams:
            issues.append(
                GraphValidationIssue("error", "empty_graph", "graph has no teams")
            )
        for team in graph.teams:
            if team.id in teams_by_id:
                issues.append(
                    GraphValidationIssue(
                        "error",
                        "duplicate_team_id",
                        f"team ID {team.id} appears more than once",
                        team_id=team.id,
                    )
                )
            else:
                teams_by_id[team.id] = team
        return teams_by_id

    @staticmethod
    def _validate_roots(
        graph: GraphLike,
        teams_by_id: dict[int, TeamLike],
        issues: list[GraphValidationIssue],
    ) -> None:
        if not graph.root_team_ids:
            issues.append(
                GraphValidationIssue("error", "missing_root", "graph has no roots")
            )
        seen: set[int] = set()
        for root_id in graph.root_team_ids:
            if root_id in seen:
                issues.append(
                    GraphValidationIssue(
                        "error",
                        "duplicate_root_id",
                        f"root team ID {root_id} appears more than once",
                        team_id=root_id,
                    )
                )
            elif root_id not in teams_by_id:
                issues.append(
                    GraphValidationIssue(
                        "error",
                        "missing_root_team",
                        f"root team ID {root_id} does not resolve",
                        team_id=root_id,
                    )
                )
            seen.add(root_id)

    @classmethod
    def _validate_team(
        cls,
        team: TeamLike,
        teams_by_id: dict[int, TeamLike],
        executor: ProgramExecutor,
        learners_by_id: dict[int, LearnerLike],
        programs_by_id: dict[int, ProgramLike],
        issues: list[GraphValidationIssue],
    ) -> None:
        if not team.learners:
            issues.append(
                GraphValidationIssue(
                    "error",
                    "empty_team",
                    f"team {team.id} has no learners",
                    team_id=team.id,
                )
            )
            return

        if not any(learner.action.kind == "atomic" for learner in team.learners):
            issues.append(
                GraphValidationIssue(
                    "error",
                    "team_without_atomic_action",
                    f"team {team.id} has no atomic-action learner",
                    team_id=team.id,
                )
            )

        local_learner_ids: set[int] = set()
        for learner in team.learners:
            if learner.id in local_learner_ids:
                issues.append(
                    GraphValidationIssue(
                        "error",
                        "duplicate_team_learner",
                        f"team {team.id} repeats learner ID {learner.id}",
                        team_id=team.id,
                        learner_id=learner.id,
                    )
                )
            local_learner_ids.add(learner.id)
            cls._validate_learner_identity(learner, learners_by_id, issues, team.id)
            cls._validate_program(
                learner,
                executor,
                programs_by_id,
                issues,
                team.id,
            )
            cls._validate_action(learner, team.id, teams_by_id, issues)

    @staticmethod
    def _validate_learner_identity(
        learner: LearnerLike,
        learners_by_id: dict[int, LearnerLike],
        issues: list[GraphValidationIssue],
        team_id: int,
    ) -> None:
        existing = learners_by_id.get(learner.id)
        if existing is not None and existing != learner:
            issues.append(
                GraphValidationIssue(
                    "error",
                    "inconsistent_learner_id",
                    f"learner ID {learner.id} denotes unequal values",
                    team_id=team_id,
                    learner_id=learner.id,
                )
            )
        learners_by_id[learner.id] = learner

    @staticmethod
    def _validate_program(
        learner: LearnerLike,
        executor: ProgramExecutor,
        programs_by_id: dict[int, ProgramLike],
        issues: list[GraphValidationIssue],
        team_id: int,
    ) -> None:
        program = learner.program
        existing = programs_by_id.get(program.id)
        if existing is not None and existing != program:
            issues.append(
                GraphValidationIssue(
                    "error",
                    "inconsistent_program_id",
                    f"program ID {program.id} denotes unequal values",
                    team_id=team_id,
                    learner_id=learner.id,
                    program_id=program.id,
                )
            )
        programs_by_id[program.id] = program

        if not program.instructions:
            issues.append(
                GraphValidationIssue(
                    "error",
                    "empty_program",
                    f"program {program.id} has no instructions",
                    team_id=team_id,
                    learner_id=learner.id,
                    program_id=program.id,
                )
            )
        for position, instruction in enumerate(program.instructions):
            try:
                executor.validate_instruction(instruction, position)
            except InvalidInstructionError as error:
                if isinstance(error, UnknownOperatorError):
                    code = "unknown_operator"
                elif isinstance(error, OperatorArityError):
                    code = "operator_arity"
                else:
                    code = "invalid_instruction"
                issues.append(
                    GraphValidationIssue(
                        "error",
                        code,
                        str(error),
                        team_id=team_id,
                        learner_id=learner.id,
                        program_id=program.id,
                        instruction_index=position,
                    )
                )

    @staticmethod
    def _validate_action(
        learner: LearnerLike,
        team_id: int,
        teams_by_id: dict[int, TeamLike],
        issues: list[GraphValidationIssue],
    ) -> None:
        if learner.action.kind == "atomic":
            return
        target = learner.action.team_id
        if target == team_id:
            issues.append(
                GraphValidationIssue(
                    "error",
                    "self_reference",
                    f"learner {learner.id} references its own team {team_id}",
                    team_id=team_id,
                    learner_id=learner.id,
                )
            )
        if target not in teams_by_id:
            issues.append(
                GraphValidationIssue(
                    "error",
                    "dangling_team_reference",
                    f"learner {learner.id} references missing team {target}",
                    team_id=team_id,
                    learner_id=learner.id,
                )
            )

    @staticmethod
    def _report_orphans(
        graph: GraphLike,
        teams_by_id: dict[int, TeamLike],
        issues: list[GraphValidationIssue],
    ) -> None:
        reachable: set[int] = set()
        pending = [root for root in graph.root_team_ids if root in teams_by_id]
        while pending:
            team_id = pending.pop()
            if team_id in reachable:
                continue
            reachable.add(team_id)
            for learner in teams_by_id[team_id].learners:
                if (
                    learner.action.kind == "team_reference"
                    and learner.action.team_id in teams_by_id
                ):
                    pending.append(learner.action.team_id)

        for team_id in sorted(teams_by_id.keys() - reachable):
            issues.append(
                GraphValidationIssue(
                    "warning",
                    "orphan_team",
                    f"team {team_id} is unreachable from every root",
                    team_id=team_id,
                )
            )


__all__ = [
    "GraphValidationError",
    "GraphValidationIssue",
    "GraphValidationReport",
    "GraphValidator",
    "ValidationSeverity",
]
