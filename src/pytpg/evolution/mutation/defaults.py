"""Default mutation composition with explicit relative category weights."""

import math
from dataclasses import dataclass

from pytpg.evolution.errors import EvolutionConfigurationError
from pytpg.evolution.genome import GenomeFactory
from pytpg.evolution.mutation.base import WeightedMutation
from pytpg.evolution.mutation.instruction import (
    DeleteInstructionMutation,
    InsertInstructionMutation,
    ModifyInstructionMutation,
)
from pytpg.evolution.mutation.learner import (
    AddLearnerMutation,
    DeleteLearnerMutation,
    MutateLearnerAction,
)
from pytpg.evolution.mutation.team import AddTeamMutation, DeleteTeamMutation


@dataclass(frozen=True, slots=True)
class MutationConfig:
    """Bounds and relative weights for the reference mutation mixture."""

    min_team_size: int = 2
    max_team_size: int = 8
    new_team_size: int = 2
    instruction_insert_weight: float = 1.0
    instruction_delete_weight: float = 1.0
    instruction_modify_weight: float = 2.0
    learner_action_weight: float = 2.0
    learner_add_weight: float = 1.0
    learner_delete_weight: float = 1.0
    team_add_weight: float = 0.5
    team_delete_weight: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("min_team_size", self.min_team_size),
            ("max_team_size", self.max_team_size),
            ("new_team_size", self.new_team_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EvolutionConfigurationError(f"{name} must be a positive integer")
        if self.max_team_size < self.min_team_size:
            raise EvolutionConfigurationError(
                "max_team_size must be at least min_team_size"
            )
        if not self.min_team_size <= self.new_team_size <= self.max_team_size:
            raise EvolutionConfigurationError(
                "new_team_size must be within the configured team-size bounds"
            )
        weights = (
            self.instruction_insert_weight,
            self.instruction_delete_weight,
            self.instruction_modify_weight,
            self.learner_action_weight,
            self.learner_add_weight,
            self.learner_delete_weight,
            self.team_add_weight,
            self.team_delete_weight,
        )
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or weight < 0.0
            for weight in weights
        ):
            raise EvolutionConfigurationError(
                "mutation weights must be finite non-negative real numbers"
            )
        if sum(weights) <= 0.0:
            raise EvolutionConfigurationError(
                "at least one mutation weight must be positive"
            )


def default_mutation(
    factory: GenomeFactory,
    config: MutationConfig | None = None,
) -> WeightedMutation:
    """Construct the standard eight-category weighted mutation strategy."""

    selected = config or MutationConfig()
    program_length = min(
        max(4, factory.config.min_program_length),
        factory.config.max_program_length,
    )
    operators = (
        InsertInstructionMutation(factory),
        DeleteInstructionMutation(factory),
        ModifyInstructionMutation(factory),
        MutateLearnerAction(factory),
        AddLearnerMutation(factory, selected.max_team_size, program_length),
        DeleteLearnerMutation(factory, selected.min_team_size),
        AddTeamMutation(
            factory,
            selected.max_team_size,
            selected.new_team_size,
            program_length,
        ),
        DeleteTeamMutation(factory, selected.min_team_size),
    )
    weights = (
        selected.instruction_insert_weight,
        selected.instruction_delete_weight,
        selected.instruction_modify_weight,
        selected.learner_action_weight,
        selected.learner_add_weight,
        selected.learner_delete_weight,
        selected.team_add_weight,
        selected.team_delete_weight,
    )
    return WeightedMutation(operators, weights)


__all__ = ["MutationConfig", "default_mutation"]
