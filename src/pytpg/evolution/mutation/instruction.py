"""Focused instruction insertion, deletion, and modification operators."""

from __future__ import annotations

from dataclasses import dataclass

from pytpg.core import Learner, Program, TPGGraph
from pytpg.evolution._ids import GraphIDAllocator
from pytpg.evolution.genome import GenomeFactory
from pytpg.evolution.mutation._utils import (
    choose_index,
    noop,
    replace_learner,
    validated_outcome,
)
from pytpg.evolution.mutation.base import MutationOutcome
from pytpg.evolution.rng import RandomGenerator


def _memberships(
    graph: TPGGraph,
    *,
    minimum_length: int | None = None,
    maximum_length: int | None = None,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (team_index, learner_index)
        for team_index, team in enumerate(graph.teams)
        for learner_index, learner in enumerate(team.learners)
        if (
            minimum_length is None or len(learner.program.instructions) > minimum_length
        )
        and (
            maximum_length is None or len(learner.program.instructions) < maximum_length
        )
    )


@dataclass(frozen=True, slots=True)
class InsertInstructionMutation:
    factory: GenomeFactory
    name: str = "instruction_insert"

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome:
        candidates = _memberships(
            graph,
            maximum_length=self.factory.config.max_program_length,
        )
        if not candidates:
            return noop(graph, self.name, "no program is below maximum length")
        team_index, learner_index = candidates[choose_index(rng, len(candidates))]
        source = graph.teams[team_index].learners[learner_index]
        instructions = list(source.program.instructions)
        position = int(rng.integers(len(instructions) + 1))
        instructions.insert(position, self.factory.random_instruction(rng))
        ids = GraphIDAllocator.from_graph(graph)
        learner = Learner(
            ids.learner_id(),
            Program(ids.program_id(), tuple(instructions)),
            source.action,
        )
        candidate = replace_learner(graph, team_index, learner_index, learner)
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"inserted instruction {position} in team {graph.teams[team_index].id}",
        )


@dataclass(frozen=True, slots=True)
class DeleteInstructionMutation:
    factory: GenomeFactory
    name: str = "instruction_delete"

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome:
        candidates = _memberships(
            graph,
            minimum_length=self.factory.config.min_program_length,
        )
        if not candidates:
            return noop(graph, self.name, "no program is above minimum length")
        team_index, learner_index = candidates[choose_index(rng, len(candidates))]
        source = graph.teams[team_index].learners[learner_index]
        instructions = list(source.program.instructions)
        position = choose_index(rng, len(instructions))
        del instructions[position]
        ids = GraphIDAllocator.from_graph(graph)
        learner = Learner(
            ids.learner_id(),
            Program(ids.program_id(), tuple(instructions)),
            source.action,
        )
        candidate = replace_learner(graph, team_index, learner_index, learner)
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"deleted instruction {position} in team {graph.teams[team_index].id}",
        )


@dataclass(frozen=True, slots=True)
class ModifyInstructionMutation:
    factory: GenomeFactory
    name: str = "instruction_modify"

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome:
        candidates = _memberships(graph)
        team_index, learner_index = candidates[choose_index(rng, len(candidates))]
        source = graph.teams[team_index].learners[learner_index]
        instructions = list(source.program.instructions)
        position = choose_index(rng, len(instructions))
        instructions[position] = self.factory.random_instruction(rng)
        ids = GraphIDAllocator.from_graph(graph)
        learner = Learner(
            ids.learner_id(),
            Program(ids.program_id(), tuple(instructions)),
            source.action,
        )
        candidate = replace_learner(graph, team_index, learner_index, learner)
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"modified instruction {position} in team {graph.teams[team_index].id}",
        )


__all__ = [
    "DeleteInstructionMutation",
    "InsertInstructionMutation",
    "ModifyInstructionMutation",
]
