"""Modular reference evolution for immutable TPG graph populations."""

from tpg.evolution.engine import EvolutionEngine, EvolutionRunResult
from tpg.evolution.errors import (
    EvolutionConfigurationError,
    EvolutionError,
    InvalidFitnessError,
    MutationInvariantError,
)
from tpg.evolution.genome import GenomeConfig, GenomeFactory
from tpg.evolution.initialization import (
    GraphInitializer,
    InitializationConfig,
    PopulationInitializer,
)
from tpg.evolution.mutation import (
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
from tpg.evolution.population import (
    EvaluatedIndividual,
    EvaluatedPopulation,
    Population,
)
from tpg.evolution.reproduction import (
    ChildRecord,
    Reproducer,
    ReproductionConfig,
    ReproductionResult,
)
from tpg.evolution.rng import RandomGenerator, create_rng
from tpg.evolution.selection import SelectionStrategy, TournamentSelection

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
