"""Learner action and team-membership mutation operators."""

from __future__ import annotations

from dataclasses import dataclass

from tpg.core import (
    Action,
    ActionID,
    AtomicAction,
    Learner,
    Team,
    TeamReference,
    TPGGraph,
)
from tpg.evolution._ids import GraphIDAllocator
from tpg.evolution.errors import EvolutionConfigurationError
from tpg.evolution.genome import GenomeFactory
from tpg.evolution.mutation._utils import (
    choose_index,
    noop,
    replace_learner,
    replace_team,
    validated_outcome,
)
from tpg.evolution.mutation.base import MutationOutcome
from tpg.evolution.rng import RandomGenerator


@dataclass(frozen=True, slots=True)
class MutateLearnerAction:
    factory: GenomeFactory
    name: str = "learner_action"

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome:
        candidates: list[tuple[int, int, tuple[Action, ...]]] = []
        for team_index, team in enumerate(graph.teams):
            atomic_count = sum(
                learner.action.kind == "atomic" for learner in team.learners
            )
            for learner_index, source in enumerate(team.learners):
                actions: list[Action] = [
                    AtomicAction(ActionID(action_id))
                    for action_id in range(self.factory.config.n_actions)
                    if AtomicAction(ActionID(action_id)) != source.action
                ]
                if not (source.action.kind == "atomic" and atomic_count == 1):
                    actions.extend(
                        TeamReference(target.id)
                        for target in graph.teams
                        if target.id != team.id
                        and TeamReference(target.id) != source.action
                    )
                if actions:
                    candidates.append((team_index, learner_index, tuple(actions)))

        if not candidates:
            return noop(graph, self.name, "no alternative action preserves invariants")
        team_index, learner_index, selected_actions = candidates[
            choose_index(rng, len(candidates))
        ]
        source = graph.teams[team_index].learners[learner_index]
        action = selected_actions[choose_index(rng, len(selected_actions))]
        ids = GraphIDAllocator.from_graph(graph)
        learner = Learner(ids.learner_id(), source.program, action)
        candidate = replace_learner(graph, team_index, learner_index, learner)
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"changed learner action in team {graph.teams[team_index].id}",
        )


@dataclass(frozen=True, slots=True)
class AddLearnerMutation:
    factory: GenomeFactory
    max_team_size: int = 8
    program_length: int = 4
    name: str = "learner_add"

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_team_size, bool)
            or not isinstance(self.max_team_size, int)
            or self.max_team_size < 1
        ):
            raise EvolutionConfigurationError(
                "max_team_size must be a positive integer"
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
        candidates = tuple(
            index
            for index, team in enumerate(graph.teams)
            if len(team.learners) < self.max_team_size
        )
        if not candidates:
            return noop(graph, self.name, "all teams are at maximum size")
        team_index = candidates[choose_index(rng, len(candidates))]
        team = graph.teams[team_index]
        ids = GraphIDAllocator.from_graph(graph)
        learner = self.factory.random_atomic_learner(
            ids.learner_id(),
            ids.program_id(),
            self.program_length,
            rng,
        )
        learners = list(team.learners)
        position = int(rng.integers(len(learners) + 1))
        learners.insert(position, learner)
        candidate = replace_team(graph, team_index, Team(team.id, tuple(learners)))
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"added learner at position {position} in team {team.id}",
        )


@dataclass(frozen=True, slots=True)
class DeleteLearnerMutation:
    factory: GenomeFactory
    min_team_size: int = 2
    name: str = "learner_delete"

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
        candidates: list[tuple[int, int]] = []
        for team_index, team in enumerate(graph.teams):
            if len(team.learners) <= self.min_team_size:
                continue
            atomic_count = sum(
                learner.action.kind == "atomic" for learner in team.learners
            )
            for learner_index, learner in enumerate(team.learners):
                if learner.action.kind != "atomic" or atomic_count > 1:
                    candidates.append((team_index, learner_index))
        if not candidates:
            return noop(graph, self.name, "no learner can be deleted safely")
        team_index, learner_index = candidates[choose_index(rng, len(candidates))]
        team = graph.teams[team_index]
        learners = list(team.learners)
        removed = learners.pop(learner_index)
        candidate = replace_team(graph, team_index, Team(team.id, tuple(learners)))
        return validated_outcome(
            graph,
            candidate,
            self.factory,
            self.name,
            f"deleted learner {removed.id} from team {team.id}",
        )


__all__ = ["AddLearnerMutation", "DeleteLearnerMutation", "MutateLearnerAction"]
