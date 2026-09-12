"""Team creation and deletion mutation operators."""

from __future__ import annotations

from dataclasses import dataclass

from pytpg.core import Learner, Team, TeamReference, TPGGraph
from pytpg.evolution._ids import GraphIDAllocator
from pytpg.evolution.errors import EvolutionConfigurationError
from pytpg.evolution.genome import GenomeFactory
from pytpg.evolution.mutation._utils import (
    choose_index,
    noop,
    validated_outcome,
)
from pytpg.evolution.mutation.base import MutationOutcome
from pytpg.evolution.rng import RandomGenerator
from pytpg.runtime import reachable_team_ids


@dataclass(frozen=True, slots=True)
class AddTeamMutation:
    factory: GenomeFactory
    max_team_size: int = 8
    new_team_size: int = 2
    program_length: int = 4
    name: str = "team_add"

    def __post_init__(self) -> None:
        for name, value in (
            ("max_team_size", self.max_team_size),
            ("new_team_size", self.new_team_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EvolutionConfigurationError(
                    f"{name} must be a positive integer"
                )
        if self.new_team_size > self.max_team_size:
            raise EvolutionConfigurationError(
                "new_team_size cannot exceed max_team_size"
            )
        if (
            isinstance(self.program_length, bool)
            or not isinstance(self.program_length, int)
            or not self.factory.config.min_program_length
            <= self.program_length
            <= self.factory.config.max_program_length
        ):
            raise EvolutionConfigurationError(
                "program_length is outside the configured genome bounds"
            )

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome:
        reachable = reachable_team_ids(graph)
        parents = tuple(
            index
            for index, team in enumerate(graph.teams)
            if team.id in reachable and len(team.learners) < self.max_team_size
        )
        if not parents:
            return noop(graph, self.name, "no reachable team can accept a reference")
        parent_index = parents[choose_index(rng, len(parents))]
        parent = graph.teams[parent_index]
        ids = GraphIDAllocator.from_graph(graph)
        team_id = ids.team_id()

        new_learners = tuple(
            self.factory.random_atomic_learner(
                ids.learner_id(),
                ids.program_id(),
                self.program_length,
                rng,
            )
            for _ in range(self.new_team_size)
        )
        new_team = Team(team_id, new_learners)

        reference = Learner(
            ids.learner_id(),
            self.factory.random_program(ids.program_id(), self.program_length, rng),
            TeamReference(team_id),
        )
        parent_learners = list(parent.learners)
        position = int(rng.integers(len(parent_learners) + 1))
        parent_learners.insert(position, reference)
        teams = list(graph.teams)
        teams[parent_index] = Team(parent.id, tuple(parent_learners))
        teams.append(new_team)
        candidate = TPGGraph(tuple(teams), graph.root_team_ids)
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"added team {team_id} referenced by team {parent.id}",
        )


@dataclass(frozen=True, slots=True)
class DeleteTeamMutation:
    factory: GenomeFactory
    min_team_size: int = 2
    name: str = "team_delete"

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_team_size, bool)
            or not isinstance(self.min_team_size, int)
            or self.min_team_size < 1
        ):
            raise EvolutionConfigurationError(
                "min_team_size must be a positive integer"
            )

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome:
        root_ids = set(graph.root_team_ids)
        candidates: list[tuple[int, dict[int, tuple[Learner, ...]]]] = []
        for target_index, target in enumerate(graph.teams):
            if target.id in root_ids:
                continue
            replacements: dict[int, tuple[Learner, ...]] = {}
            legal = True
            for source_index, source in enumerate(graph.teams):
                if source_index == target_index:
                    continue
                remaining = tuple(
                    learner
                    for learner in source.learners
                    if not (
                        learner.action.kind == "team_reference"
                        and learner.action.team_id == target.id
                    )
                )
                if remaining != source.learners and len(remaining) < self.min_team_size:
                    legal = False
                    break
                if remaining != source.learners:
                    replacements[source_index] = remaining
            if legal:
                candidates.append((target_index, replacements))

        if not candidates:
            return noop(graph, self.name, "no non-root team can be deleted safely")
        target_index, replacements = candidates[choose_index(rng, len(candidates))]
        target = graph.teams[target_index]
        teams: list[Team] = []
        for source_index, source in enumerate(graph.teams):
            if source_index == target_index:
                continue
            learners = replacements.get(source_index, source.learners)
            teams.append(Team(source.id, learners))
        candidate = TPGGraph(tuple(teams), graph.root_team_ids)
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"deleted team {target.id} and its incoming references",
        )


__all__ = ["AddTeamMutation", "DeleteTeamMutation"]
