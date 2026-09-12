"""Defensive validation tests for public research-infrastructure values."""

import logging
from collections.abc import Callable
from dataclasses import replace

import pytest

from pytpg.callbacks import EvolutionEvent, LoggingCallback
from pytpg.config import TPGConfig
from pytpg.evaluation import (
    ContextualBandit,
    GenerationStatistics,
    OperatorStatistics,
    RunStatistics,
    SequentialEvaluator,
)
from pytpg.evolution import (
    EvolutionConfigurationError,
    EvolutionEngine,
    EvolutionRunResult,
    GenomeConfig,
    InitializationConfig,
    ReproductionConfig,
    create_rng,
)
from pytpg.experiment import run_experiment
from pytpg.metadata import ExperimentMetadata
from pytpg.serialization import (
    Checkpoint,
    CheckpointCompatibilityError,
    InvalidSerializedDataError,
    RNGSnapshot,
    capture_rng,
    restore_rng,
)


def config(population_size: int = 4) -> TPGConfig:
    return TPGConfig(
        InitializationConfig(
            GenomeConfig(1, 2, 2),
            population_size=population_size,
            learners_per_team=2,
            program_length=1,
        ),
        tournament_size=2,
        reproduction=ReproductionConfig(elite_count=1),
    )


@pytest.mark.parametrize(
    "event",
    [
        lambda: EvolutionEvent("unknown", 0, 1),
        lambda: EvolutionEvent("run_started", -1, 1),
        lambda: EvolutionEvent("run_started", 0, 0),
        lambda: EvolutionEvent("run_started", 0, 1, resumed=1),
        lambda: EvolutionEvent(
            "generation_evaluated", 0, 1, best_fitness=float("nan")
        ),
        lambda: EvolutionEvent(
            "generation_reproduced",
            1,
            1,
            mutation_attempts=0,
            applied_mutations=1,
        ),
    ],
)
def test_invalid_lifecycle_event_is_rejected(
    event: Callable[[], EvolutionEvent],
) -> None:
    with pytest.raises(ValueError):
        event()


def test_logging_callback_rejects_invalid_constructor_values() -> None:
    with pytest.raises(TypeError, match="logger"):
        LoggingCallback(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="level"):
        LoggingCallback(logging.getLogger("test"), True)


@pytest.mark.parametrize(
    "statistic",
    [
        lambda: OperatorStatistics("", 1, 0),
        lambda: OperatorStatistics("x", 0, 0),
        lambda: OperatorStatistics("x", 1, 2),
    ],
)
def test_invalid_operator_statistics_are_rejected(
    statistic: Callable[[], OperatorStatistics],
) -> None:
    with pytest.raises(EvolutionConfigurationError):
        statistic()


def generation_statistics(**overrides: object) -> GenerationStatistics:
    values: dict[str, object] = {
        "generation": 0,
        "population_size": 2,
        "minimum_fitness": 0.0,
        "maximum_fitness": 1.0,
        "mean_fitness": 0.5,
        "median_fitness": 0.5,
        "population_stddev": 0.5,
        "best_position": 0,
    }
    values.update(overrides)
    return GenerationStatistics(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"generation": -1},
        {"population_size": 0},
        {"best_position": 2},
        {"minimum_fitness": float("nan")},
        {"minimum_fitness": 2.0, "maximum_fitness": 1.0},
        {"mean_fitness": 2.0},
        {"population_stddev": -1.0},
        {"elite_count": 3},
        {"applied_mutations": 1},
        {"operators": []},
        {
            "mutation_attempts": 2,
            "operators": (OperatorStatistics("x", 1, 1),),
            "applied_mutations": 1,
        },
        {
            "mutation_attempts": 1,
            "operators": (OperatorStatistics("x", 1, 1),),
            "applied_mutations": 0,
        },
    ],
)
def test_invalid_generation_statistics_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(EvolutionConfigurationError):
        generation_statistics(**overrides)


def test_run_statistics_require_nonempty_consecutive_records() -> None:
    with pytest.raises(EvolutionConfigurationError, match="empty"):
        RunStatistics(())
    with pytest.raises(EvolutionConfigurationError, match="consecutive"):
        RunStatistics(
            (
                generation_statistics(generation=0),
                generation_statistics(generation=2),
            )
        )


def test_evolution_run_result_rejects_inconsistent_sequences() -> None:
    result = run_experiment(
        config(), ContextualBandit.signed_binary(), 1, 4
    ).evolution

    with pytest.raises(EvolutionConfigurationError, match="requires evaluations"):
        EvolutionRunResult((), ())
    with pytest.raises(EvolutionConfigurationError, match="one more evaluation"):
        EvolutionRunResult(result.evaluations, ())
    wrong_generation = replace(
        result.reproductions[0].population,
        generation=9,
    )
    wrong_reproduction = replace(
        result.reproductions[0], population=wrong_generation
    )
    with pytest.raises(EvolutionConfigurationError, match="generation"):
        EvolutionRunResult(result.evaluations, (wrong_reproduction,))


def test_rng_snapshot_rejects_unsupported_and_malformed_sources() -> None:
    with pytest.raises(CheckpointCompatibilityError, match="unsupported"):
        RNGSnapshot("MT19937", "{}")
    with pytest.raises(InvalidSerializedDataError, match="valid JSON"):
        RNGSnapshot("PCG64", "not-json")
    with pytest.raises(InvalidSerializedDataError, match="JSON object"):
        RNGSnapshot("PCG64", "[]")
    with pytest.raises(CheckpointCompatibilityError, match="expose"):
        capture_rng(object())  # type: ignore[arg-type]
    with pytest.raises(CheckpointCompatibilityError, match="RNGSnapshot"):
        restore_rng(object())  # type: ignore[arg-type]


def test_checkpoint_rejects_population_and_statistics_boundary_mismatch() -> None:
    source_config = config()
    result = run_experiment(
        source_config, ContextualBandit.signed_binary(), 0, 4
    )
    larger_config = config(population_size=5)
    larger_metadata = ExperimentMetadata.capture(
        "larger",
        4,
        larger_config,
        created_at_utc="2026-08-29T12:00:00+00:00",
    )
    with pytest.raises(CheckpointCompatibilityError, match="population size"):
        Checkpoint(
            larger_config,
            larger_metadata,
            result.evolution.evaluations[-1],
            result.rng,
            result.statistics,
        )

    wrong_statistics = RunStatistics(
        (replace(result.statistics.generations[-1], generation=1),)
    )
    with pytest.raises(CheckpointCompatibilityError, match="statistics"):
        Checkpoint(
            source_config,
            result.metadata,
            result.evolution.evaluations[-1],
            result.rng,
            wrong_statistics,
        )


def test_experiment_rejects_supplied_metadata_for_another_seed_or_config() -> None:
    source_config = config()
    wrong_seed = ExperimentMetadata.capture(
        "wrong",
        5,
        source_config,
        created_at_utc="2026-08-29T12:00:00+00:00",
    )
    with pytest.raises(CheckpointCompatibilityError, match="master seed"):
        run_experiment(
            source_config,
            ContextualBandit.signed_binary(),
            0,
            4,
            metadata=wrong_seed,
        )

    other_config = config(population_size=5)
    wrong_config = ExperimentMetadata.capture(
        "wrong",
        4,
        other_config,
        created_at_utc="2026-08-29T12:00:00+00:00",
    )
    with pytest.raises(CheckpointCompatibilityError, match="digest"):
        run_experiment(
            source_config,
            ContextualBandit.signed_binary(),
            0,
            4,
            metadata=wrong_config,
        )


def test_invalid_resume_generation_count_is_rejected() -> None:
    result = run_experiment(
        config(), ContextualBandit.signed_binary(), 0, 4
    )
    initializer = result.config.create_initializer()
    reproducer = result.config.create_reproducer(initializer.factory)

    assert reproducer.config == result.config.reproduction
    with pytest.raises(EvolutionConfigurationError, match="generations"):
        EvolutionEngine(SequentialEvaluator(), reproducer).resume(
            result.evolution.evaluations[-1],
            ContextualBandit.signed_binary(),
            -1,
            create_rng(1),
        )
