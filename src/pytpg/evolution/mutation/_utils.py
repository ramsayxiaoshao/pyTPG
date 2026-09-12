"""Shared immutable reconstruction and validation helpers for mutations."""

from __future__ import annotations

from pytpg.core import Learner, Team, TPGGraph
from pytpg.evolution.errors import MutationInvariantError
from pytpg.evolution.genome import GenomeFactory
from pytpg.evolution.mutation.base import MutationOutcome
from pytpg.evolution.rng import RandomGenerator


def choose_index(rng: RandomGenerator, length: int) -> int:
    """Choose one valid sequence index using the explicit RNG."""

    return int(rng.integers(length))


def replace_team(graph: TPGGraph, team_index: int, team: Team) -> TPGGraph:
    """Return a graph with one team value replaced in stable order."""

    teams = list(graph.teams)
    teams[team_index] = team
    return TPGGraph(tuple(teams), graph.root_team_ids)


def replace_learner(
    graph: TPGGraph,
    team_index: int,
    learner_index: int,
    learner: Learner,
) -> TPGGraph:
    """Clone one team membership while leaving all shared values untouched."""

    team = graph.teams[team_index]
    learners = list(team.learners)
    learners[learner_index] = learner
    return replace_team(graph, team_index, Team(team.id, tuple(learners)))


def validated_outcome(
    original: TPGGraph,
    candidate: TPGGraph,
    factory: GenomeFactory,
    operator: str,
    description: str,
) -> MutationOutcome:
    """Reject invalid operator implementations before they enter a population."""

    report = candidate.validate(factory.runtime_config, operators=factory.operators)
    if not report.is_valid:
        codes = ", ".join(issue.code for issue in report.errors)
        raise MutationInvariantError(f"{operator} produced invalid graph: {codes}")
    return MutationOutcome(candidate, operator, candidate != original, description)


def noop(graph: TPGGraph, operator: str, reason: str) -> MutationOutcome:
    """Return an explicit non-applied mutation attempt."""

    return MutationOutcome(graph, operator, False, reason)
