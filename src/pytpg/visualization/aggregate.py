"""Whole-session usage, including repeated evaluations of shared programs."""

from collections import Counter
from dataclasses import dataclass

from pytpg.visualization.model import TraceSession


@dataclass(frozen=True, slots=True)
class ActivationCounts:
    team_visits: Counter[int]
    program_evaluations: Counter[int]
    program_wins: Counter[int]
    learner_evaluations: Counter[tuple[int, int]]
    learner_wins: Counter[tuple[int, int]]
    edge_traversals: Counter[tuple[int, int]]
    atomic_actions: Counter[int]


def aggregate(session: TraceSession) -> ActivationCounts:
    """Count events; edge keys identify each selected learner's destination edge."""
    counts = ActivationCounts(
        Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    )
    for frame in session.frames:
        counts.atomic_actions[frame.traversal.action_id] += 1
        for decision in frame.traversal.team_decisions:
            counts.team_visits[decision.team_id] += 1
            for item in decision.evaluations:
                key = decision.team_id, item.learner_id
                if item.evaluated:
                    counts.program_evaluations[item.program_id] += 1
                    counts.learner_evaluations[key] += 1
                if item.winner:
                    counts.program_wins[item.program_id] += 1
                    counts.learner_wins[key] += 1
                    counts.edge_traversals[key] += 1
    return counts
