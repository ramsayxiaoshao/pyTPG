"""Deterministic graph-local ID allocation with no global counters."""

from __future__ import annotations

from dataclasses import dataclass

from pytpg.core import LearnerID, ProgramID, TeamID, TPGGraph


@dataclass(slots=True)
class GraphIDAllocator:
    """Allocate monotonically increasing IDs above one graph's current maxima."""

    next_team: int
    next_learner: int
    next_program: int

    @classmethod
    def from_graph(cls, graph: TPGGraph) -> GraphIDAllocator:
        """Initialize counters solely from explicit graph contents."""

        team_ids = [int(team.id) for team in graph.teams]
        learner_ids = [
            int(learner.id) for team in graph.teams for learner in team.learners
        ]
        program_ids = [
            int(learner.program.id) for team in graph.teams for learner in team.learners
        ]
        return cls(
            next_team=max(team_ids, default=-1) + 1,
            next_learner=max(learner_ids, default=-1) + 1,
            next_program=max(program_ids, default=-1) + 1,
        )

    def team_id(self) -> TeamID:
        value = TeamID(self.next_team)
        self.next_team += 1
        return value

    def learner_id(self) -> LearnerID:
        value = LearnerID(self.next_learner)
        self.next_learner += 1
        return value

    def program_id(self) -> ProgramID:
        value = ProgramID(self.next_program)
        self.next_program += 1
        return value


__all__ = ["GraphIDAllocator"]
