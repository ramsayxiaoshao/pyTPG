"""Deterministic initialization of valid graph populations."""

from __future__ import annotations

from dataclasses import dataclass

from pytpg.core import (
    Learner,
    LearnerID,
    ProgramID,
    Team,
    TeamID,
    TeamReference,
    TPGGraph,
)
from pytpg.evolution.errors import EvolutionConfigurationError, MutationInvariantError
from pytpg.evolution.genome import GenomeConfig, GenomeFactory
from pytpg.evolution.population import Population
from pytpg.evolution.rng import RandomGenerator
from pytpg.runtime import OperatorRegistry


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    """Population topology and initial program-size policy."""

    genome: GenomeConfig
    population_size: int = 32
    team_count: int = 1
    learners_per_team: int = 4
    program_length: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("population_size", self.population_size),
            ("team_count", self.team_count),
            ("learners_per_team", self.learners_per_team),
            ("program_length", self.program_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                msg = f"{name} must be a positive integer, got {value!r}"
                raise EvolutionConfigurationError(msg)
        if self.learners_per_team < 2:
            raise EvolutionConfigurationError(
                "learners_per_team must be at least 2 for stable graph invariants"
            )
        if not (
            self.genome.min_program_length
            <= self.program_length
            <= self.genome.max_program_length
        ):
            raise EvolutionConfigurationError(
                "program_length is outside the configured genome bounds"
            )


class GraphInitializer:
    """Create connected valid graphs with a single explicit root."""

    __slots__ = ("config", "factory")

    def __init__(
        self,
        config: InitializationConfig,
        operators: OperatorRegistry | None = None,
    ) -> None:
        self.config = config
        self.factory = GenomeFactory(config.genome, operators)

    def initialize(self, rng: RandomGenerator) -> TPGGraph:
        """Create an atomic graph or a forward chain of reachable teams."""

        learner_id = 0
        program_id = 0
        teams: list[Team] = []
        for team_index in range(self.config.team_count):
            learners: list[Learner] = []
            for _ in range(self.config.learners_per_team):
                learners.append(
                    self.factory.random_atomic_learner(
                        LearnerID(learner_id),
                        ProgramID(program_id),
                        self.config.program_length,
                        rng,
                    )
                )
                learner_id += 1
                program_id += 1
            if team_index + 1 < self.config.team_count:
                atomic = learners[0]
                learners[0] = Learner(
                    atomic.id,
                    atomic.program,
                    TeamReference(TeamID(team_index + 1)),
                )
            teams.append(Team(TeamID(team_index), tuple(learners)))

        graph = TPGGraph(tuple(teams), (TeamID(0),))
        report = graph.validate(
            self.factory.runtime_config,
            operators=self.factory.operators,
        )
        if not report.is_valid:
            codes = ", ".join(issue.code for issue in report.errors)
            raise MutationInvariantError(f"initializer produced invalid graph: {codes}")
        return graph


class PopulationInitializer:
    """Create a stable ordered population from one graph initializer."""

    __slots__ = ("graph_initializer",)

    def __init__(self, graph_initializer: GraphInitializer) -> None:
        self.graph_initializer = graph_initializer

    def initialize(self, rng: RandomGenerator) -> Population:
        """Create generation zero using one shared explicit RNG stream."""

        return Population(
            generation=0,
            individuals=tuple(
                self.graph_initializer.initialize(rng)
                for _ in range(self.graph_initializer.config.population_size)
            ),
        )


__all__ = ["GraphInitializer", "InitializationConfig", "PopulationInitializer"]
