"""Checkpoint round-trip and uninterrupted-versus-resumed regression tests."""

from pathlib import Path

import pytest

from tpg.callbacks import EventRecorder
from tpg.config import TPGConfig
from tpg.core import TPGGraph
from tpg.evaluation import ContextualBandit
from tpg.evolution import (
    GenomeConfig,
    InitializationConfig,
    ReproductionConfig,
    create_rng,
)
from tpg.experiment import resume_experiment, run_experiment
from tpg.metadata import ExperimentMetadata
from tpg.serialization import (
    CheckpointCompatibilityError,
    InvalidSerializedDataError,
    UnsupportedFormatVersionError,
    capture_rng,
    checkpoint_from_dict,
    checkpoint_to_dict,
    dumps_checkpoint,
    load_checkpoint,
    loads_checkpoint,
    restore_rng,
    save_checkpoint,
)


def config() -> TPGConfig:
    return TPGConfig(
        InitializationConfig(
            GenomeConfig(1, 4, 2, max_program_length=8),
            population_size=6,
            learners_per_team=2,
            program_length=2,
        ),
        tournament_size=2,
        reproduction=ReproductionConfig(elite_count=1, mutation_steps=2),
    )


def metadata(source: TPGConfig) -> ExperimentMetadata:
    return ExperimentMetadata.capture(
        "resume-regression",
        42,
        source,
        created_at_utc="2026-08-29T12:00:00+00:00",
    )


def test_rng_snapshot_restores_the_exact_next_draw() -> None:
    rng = create_rng(123)
    tuple(int(rng.integers(1000)) for _ in range(7))
    snapshot = capture_rng(rng)

    expected = tuple(int(rng.integers(1_000_000)) for _ in range(20))
    restored = restore_rng(snapshot)

    assert tuple(int(restored.integers(1_000_000)) for _ in range(20)) == expected


def test_checkpoint_json_and_file_round_trip(tmp_path: Path) -> None:
    source_config = config()
    partial = run_experiment(
        source_config,
        ContextualBandit.signed_binary(),
        generations=3,
        master_seed=42,
        metadata=metadata(source_config),
    )
    checkpoint = partial.checkpoint()

    restored = loads_checkpoint(dumps_checkpoint(checkpoint))
    path = tmp_path / "checkpoints" / "generation-3.json"
    save_checkpoint(checkpoint, path)

    assert restored == checkpoint
    assert load_checkpoint(path) == checkpoint
    assert not path.read_bytes().startswith(b"\x80")


def test_resumed_run_matches_uninterrupted_population_rng_and_statistics() -> None:
    source_config = config()
    task = ContextualBandit.signed_binary()
    recorded = metadata(source_config)
    full = run_experiment(
        source_config,
        task,
        generations=5,
        master_seed=42,
        metadata=recorded,
    )
    partial = run_experiment(
        source_config,
        task,
        generations=3,
        master_seed=42,
        metadata=recorded,
    )
    resumed = resume_experiment(partial.checkpoint(), task, generations=2)

    assert resumed.evolution.final_population == full.evolution.final_population
    assert resumed.evolution.best == full.evolution.best
    assert resumed.rng == full.rng
    assert resumed.statistics == full.statistics


def test_resume_does_not_re_evaluate_checkpoint_generation() -> None:
    source_config = config()
    task = ContextualBandit.signed_binary()
    partial = run_experiment(
        source_config,
        task,
        generations=2,
        master_seed=42,
        metadata=metadata(source_config),
    )
    calls: list[TPGGraph] = []

    def counted_fitness(graph: TPGGraph) -> float:
        calls.append(graph)
        return task(graph)

    recorder = EventRecorder()
    resumed = resume_experiment(
        partial.checkpoint(),
        counted_fitness,
        generations=2,
        callbacks=(recorder,),
    )

    assert len(calls) == 2 * source_config.initialization.population_size
    assert resumed.evolution.evaluations[0] == partial.evolution.evaluations[-1]
    assert all(event.resumed for event in recorder.events)


def test_checkpoint_reader_rejects_version_and_digest_mismatch() -> None:
    source_config = config()
    result = run_experiment(
        source_config,
        ContextualBandit.signed_binary(),
        generations=0,
        master_seed=42,
        metadata=metadata(source_config),
    )
    version = checkpoint_to_dict(result.checkpoint())
    version["format_version"] = 2
    with pytest.raises(UnsupportedFormatVersionError, match="version 2"):
        checkpoint_from_dict(version)

    mismatch = checkpoint_to_dict(result.checkpoint())
    checkpoint = mismatch["checkpoint"]  # type: ignore[assignment]
    checkpoint["metadata"]["config_digest"] = "0" * 64  # type: ignore[index]
    with pytest.raises(CheckpointCompatibilityError, match="digest"):
        checkpoint_from_dict(mismatch)

    invalid_rng = checkpoint_to_dict(result.checkpoint())
    checkpoint = invalid_rng["checkpoint"]  # type: ignore[assignment]
    checkpoint["rng"]["state"] = {}  # type: ignore[index]
    with pytest.raises(InvalidSerializedDataError, match="PCG64"):
        checkpoint_from_dict(invalid_rng)

    invalid_statistics = checkpoint_to_dict(result.checkpoint())
    checkpoint = invalid_statistics["checkpoint"]  # type: ignore[assignment]
    final_statistics = checkpoint["statistics"][-1]  # type: ignore[index]
    final_statistics["best_position"] = (  # type: ignore[index]
        final_statistics["best_position"] + 1  # type: ignore[index,operator]
    ) % source_config.initialization.population_size
    with pytest.raises(CheckpointCompatibilityError, match="final statistics"):
        checkpoint_from_dict(invalid_statistics)


def test_same_recorded_inputs_produce_equal_experiment_outputs() -> None:
    source_config = config()
    recorded = metadata(source_config)

    first = run_experiment(
        source_config,
        ContextualBandit.signed_binary(),
        generations=4,
        master_seed=42,
        metadata=recorded,
    )
    second = run_experiment(
        source_config,
        ContextualBandit.signed_binary(),
        generations=4,
        master_seed=42,
        metadata=recorded,
    )

    assert first == second
