"""Lifecycle callback ordering, logging, and run-statistics tests."""

import io
import logging

import pytest

from pytpg.callbacks import EventRecorder, LoggingCallback
from pytpg.config import TPGConfig
from pytpg.evaluation import ContextualBandit
from pytpg.evolution import (
    GenomeConfig,
    InitializationConfig,
    ReproductionConfig,
)
from pytpg.experiment import run_experiment


def config() -> TPGConfig:
    return TPGConfig(
        InitializationConfig(
            GenomeConfig(1, 3, 2, max_program_length=8),
            population_size=6,
            learners_per_team=2,
            program_length=2,
        ),
        tournament_size=2,
        reproduction=ReproductionConfig(elite_count=1, mutation_steps=2),
    )


def test_callbacks_receive_complete_stable_lifecycle_order() -> None:
    recorder = EventRecorder()

    result = run_experiment(
        config(),
        ContextualBandit.signed_binary(),
        generations=2,
        master_seed=3,
        callbacks=(recorder,),
    )

    assert tuple(event.kind for event in recorder.events) == (
        "run_started",
        "generation_evaluated",
        "generation_reproduced",
        "generation_evaluated",
        "generation_reproduced",
        "generation_evaluated",
        "run_finished",
    )
    assert tuple(event.generation for event in recorder.events) == (
        0,
        0,
        1,
        1,
        2,
        2,
        2,
    )
    assert not any(event.resumed for event in recorder.events)
    reproduced = tuple(
        event for event in recorder.events if event.kind == "generation_reproduced"
    )
    assert all(event.elite_count == 1 for event in reproduced)
    assert all(event.mutation_attempts == 10 for event in reproduced)
    assert recorder.events[-1].best_fitness == result.evolution.best.fitness


def test_logging_callback_emits_structured_json_without_global_configuration() -> None:
    stream = io.StringIO()
    logger = logging.Logger("pytpg-test")
    logger.addHandler(logging.StreamHandler(stream))

    run_experiment(
        config(),
        ContextualBandit.signed_binary(),
        generations=0,
        master_seed=4,
        callbacks=(LoggingCallback(logger),),
    )

    messages = stream.getvalue().splitlines()
    assert len(messages) == 3
    assert '"kind":"run_started"' in messages[0]
    assert '"kind":"generation_evaluated"' in messages[1]
    assert '"kind":"run_finished"' in messages[2]


def test_run_statistics_cover_fitness_and_mutation_attempts() -> None:
    result = run_experiment(
        config(),
        ContextualBandit.signed_binary(),
        generations=2,
        master_seed=8,
    )

    statistics = result.statistics.generations
    assert tuple(item.generation for item in statistics) == (0, 1, 2)
    assert all(item.population_size == 6 for item in statistics)
    assert all(
        item.minimum_fitness <= item.mean_fitness <= item.maximum_fitness
        for item in statistics
    )
    assert tuple(item.mutation_attempts for item in statistics) == (10, 10, 0)
    assert tuple(item.elite_count for item in statistics) == (1, 1, 0)
    assert all(
        sum(operator.attempts for operator in item.operators)
        == item.mutation_attempts
        for item in statistics
    )


def test_callback_errors_propagate_instead_of_silently_losing_logs() -> None:
    class FailingCallback:
        def on_event(self, event: object) -> None:
            raise RuntimeError("logging unavailable")

    with pytest.raises(RuntimeError, match="logging unavailable"):
        run_experiment(
            config(),
            ContextualBandit.signed_binary(),
            generations=1,
            master_seed=9,
            callbacks=(FailingCallback(),),  # type: ignore[arg-type]
        )
