"""Immutable records captured from the decision-making execution itself."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LearnerEvaluationTrace:
    """One learner in a visited team; excluded references have no bid."""

    learner_id: int
    program_id: int
    bid: float | None
    evaluated: bool
    eligible: bool
    winner: bool
    action_kind: Literal["atomic", "team_reference"]
    atomic_action_id: int | None = None
    referenced_team_id: int | None = None


@dataclass(frozen=True, slots=True)
class TeamDecisionTrace:
    """All learners in stable team order, including excluded references."""

    team_id: int
    evaluations: tuple[LearnerEvaluationTrace, ...]

    @property
    def winner(self) -> LearnerEvaluationTrace:
        return next(item for item in self.evaluations if item.winner)

    @property
    def winner_learner_id(self) -> int:
        return self.winner.learner_id

    @property
    def winner_program_id(self) -> int:
        return self.winner.program_id

    @property
    def winner_bid(self) -> float:
        bid = self.winner.bid
        assert bid is not None
        return bid


@dataclass(frozen=True, slots=True)
class DetailedTraversalResult:
    """One complete decision, with bids from normal program execution."""

    root_team_id: int
    action_id: int
    team_decisions: tuple[TeamDecisionTrace, ...]

    @property
    def visited_team_ids(self) -> tuple[int, ...]:
        return tuple(item.team_id for item in self.team_decisions)
