"""High-level reproducible run and exact evaluated-boundary resume helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tpg.callbacks import EvolutionCallback
from tpg.config import TPGConfig
from tpg.evaluation import Evaluator, FitnessFunction, SequentialEvaluator
from tpg.evaluation.statistics import RunStatistics
from tpg.evolution import EvolutionEngine, EvolutionRunResult, PopulationInitializer
from tpg.metadata import ExperimentMetadata
from tpg.seed import SeedManager
from tpg.serialization.checkpoint import Checkpoint
from tpg.serialization.errors import CheckpointCompatibilityError
from tpg.serialization.rng import RNGSnapshot, capture_rng, restore_rng


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Evolution output plus the provenance needed to save its next boundary."""

    config: TPGConfig
    metadata: ExperimentMetadata
    evolution: EvolutionRunResult
    statistics: RunStatistics
    rng: RNGSnapshot

    def checkpoint(self) -> Checkpoint:
        """Create an immutable checkpoint without another random draw."""

        return Checkpoint(
            config=self.config,
            metadata=self.metadata,
            evaluated=self.evolution.evaluations[-1],
            rng=self.rng,
            statistics=self.statistics,
        )


def run_experiment(
    config: TPGConfig,
    fitness_function: FitnessFunction,
    generations: int,
    master_seed: int,
    *,
    name: str = "experiment",
    tags: tuple[tuple[str, str], ...] = (),
    metadata: ExperimentMetadata | None = None,
    evaluator: Evaluator | None = None,
    callbacks: Iterable[EvolutionCallback] = (),
) -> ExperimentResult:
    """Initialize and run with independent initialization/evolution streams."""

    seeds = SeedManager(master_seed)
    initializer = config.create_initializer()
    population = PopulationInitializer(initializer).initialize(
        seeds.rng("initialization")
    )
    evolution_rng = seeds.rng("evolution")
    recorded = metadata or ExperimentMetadata.capture(
        name,
        master_seed,
        config,
        tags=tags,
    )
    _require_metadata_compatibility(recorded, config, master_seed)
    engine = EvolutionEngine(
        evaluator or SequentialEvaluator(),
        config.create_reproducer(initializer.factory),
    )
    evolution = engine.run(
        population,
        fitness_function,
        generations,
        evolution_rng,
        callbacks=callbacks,
    )
    return ExperimentResult(
        config=config,
        metadata=recorded,
        evolution=evolution,
        statistics=RunStatistics.from_run(evolution),
        rng=capture_rng(evolution_rng),
    )


def resume_experiment(
    checkpoint: Checkpoint,
    fitness_function: FitnessFunction,
    generations: int,
    *,
    evaluator: Evaluator | None = None,
    callbacks: Iterable[EvolutionCallback] = (),
) -> ExperimentResult:
    """Continue exactly from saved fitness and RNG state without re-evaluation."""

    initializer = checkpoint.config.create_initializer()
    engine = EvolutionEngine(
        evaluator or SequentialEvaluator(),
        checkpoint.config.create_reproducer(initializer.factory),
    )
    evolution_rng = restore_rng(checkpoint.rng)
    evolution = engine.resume(
        checkpoint.evaluated,
        fitness_function,
        generations,
        evolution_rng,
        callbacks=callbacks,
    )
    chunk_statistics = RunStatistics.from_run(evolution)
    combined_statistics = RunStatistics(
        checkpoint.statistics.generations[:-1] + chunk_statistics.generations
    )
    return ExperimentResult(
        config=checkpoint.config,
        metadata=checkpoint.metadata,
        evolution=evolution,
        statistics=combined_statistics,
        rng=capture_rng(evolution_rng),
    )


def _require_metadata_compatibility(
    metadata: ExperimentMetadata,
    config: TPGConfig,
    master_seed: int,
) -> None:
    if metadata.config_digest != config.digest:
        raise CheckpointCompatibilityError(
            "metadata config digest does not match experiment config"
        )
    if metadata.master_seed != master_seed:
        raise CheckpointCompatibilityError(
            "metadata master seed does not match experiment seed"
        )


__all__ = ["ExperimentResult", "resume_experiment", "run_experiment"]
