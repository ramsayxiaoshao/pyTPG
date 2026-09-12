"""Modular reference evolution for immutable TPG graph populations."""

from pytpg.evolution.engine import EvolutionEngine, EvolutionRunResult
from pytpg.evolution.errors import (
    EvolutionConfigurationError,
    EvolutionError,
    InvalidFitnessError,
    MutationInvariantError,
)
from pytpg.evolution.genome import GenomeConfig, GenomeFactory
from pytpg.evolution.initialization import (
    GraphInitializer,
    InitializationConfig,
    PopulationInitializer,
)
from pytpg.evolution.mutation import (
    AddLearnerMutation,
    AddTeamMutation,
    DeleteInstructionMutation,
    DeleteLearnerMutation,
    DeleteTeamMutation,
    InsertInstructionMutation,
    ModifyInstructionMutation,
    MutateLearnerAction,
    MutationConfig,
    MutationOperator,
    MutationOutcome,
    WeightedMutation,
    default_mutation,
)
from pytpg.evolution.population import (
    EvaluatedIndividual,
    EvaluatedPopulation,
    Population,
)
from pytpg.evolution.reproduction import (
    ChildRecord,
    Reproducer,
    ReproductionConfig,
    ReproductionResult,
)
from pytpg.evolution.rng import RandomGenerator, create_rng
from pytpg.evolution.selection import SelectionStrategy, TournamentSelection

__all__ = [
    "AddLearnerMutation",
    "AddTeamMutation",
    "ChildRecord",
    "DeleteInstructionMutation",
    "DeleteLearnerMutation",
    "DeleteTeamMutation",
    "EvaluatedIndividual",
    "EvaluatedPopulation",
    "EvolutionConfigurationError",
    "EvolutionEngine",
    "EvolutionError",
    "EvolutionRunResult",
    "GenomeConfig",
    "GenomeFactory",
    "GraphInitializer",
    "InitializationConfig",
    "InsertInstructionMutation",
    "InvalidFitnessError",
    "ModifyInstructionMutation",
    "MutateLearnerAction",
    "MutationConfig",
    "MutationInvariantError",
    "MutationOperator",
    "MutationOutcome",
    "Population",
    "PopulationInitializer",
    "RandomGenerator",
    "Reproducer",
    "ReproductionConfig",
    "ReproductionResult",
    "SelectionStrategy",
    "TournamentSelection",
    "WeightedMutation",
    "create_rng",
    "default_mutation",
]
