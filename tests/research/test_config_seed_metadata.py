"""Experiment configuration, named seed, and provenance tests."""

import pytest

from pytpg.config import TPGConfig
from pytpg.evolution import (
    EvolutionConfigurationError,
    GenomeConfig,
    InitializationConfig,
    MutationConfig,
    ReproductionConfig,
)
from pytpg.metadata import ExperimentMetadata
from pytpg.seed import SeedManager


def config(**overrides: object) -> TPGConfig:
    values: dict[str, object] = {
        "initialization": InitializationConfig(
            GenomeConfig(1, 3, 2, max_program_length=8),
            population_size=6,
            learners_per_team=2,
            program_length=2,
        ),
        "mutation": MutationConfig(),
        "tournament_size": 2,
        "reproduction": ReproductionConfig(elite_count=1, mutation_steps=2),
    }
    values.update(overrides)
    return TPGConfig(**values)  # type: ignore[arg-type]


def test_config_round_trip_and_digest_are_stable() -> None:
    source = config()

    restored = TPGConfig.from_dict(source.to_dict())

    assert restored == source
    assert restored.digest == source.digest
    assert len(source.digest) == 64


def test_digest_changes_when_algorithm_configuration_changes() -> None:
    first = config()
    second = config(tournament_size=3)

    assert first.digest != second.digest


@pytest.mark.parametrize("size", [0, 7, True])
def test_invalid_tournament_size_is_rejected(size: object) -> None:
    with pytest.raises(EvolutionConfigurationError, match="tournament_size"):
        config(tournament_size=size)


def test_initial_team_size_must_fit_mutation_bounds() -> None:
    with pytest.raises(EvolutionConfigurationError, match="team-size bounds"):
        config(mutation=MutationConfig(min_team_size=3, new_team_size=3))


def test_config_reader_rejects_unknown_fields() -> None:
    data = config().to_dict()
    data["unknown"] = 1

    with pytest.raises(EvolutionConfigurationError, match="fields"):
        TPGConfig.from_dict(data)


def test_config_reader_requires_complete_nested_config() -> None:
    data = config().to_dict()
    mutation = data["mutation"]  # type: ignore[assignment]
    del mutation["team_add_weight"]  # type: ignore[index]

    with pytest.raises(EvolutionConfigurationError, match="mutation fields"):
        TPGConfig.from_dict(data)


def test_named_seed_streams_are_stable_and_request_order_independent() -> None:
    manager = SeedManager(9876)
    evolution_seed = manager.derive_seed("evolution")

    manager.rng("initialization")
    first = manager.rng("evolution")
    second = SeedManager(9876).rng("evolution")

    assert evolution_seed == SeedManager(9876).derive_seed("evolution")
    assert tuple(int(first.integers(1_000_000)) for _ in range(10)) == tuple(
        int(second.integers(1_000_000)) for _ in range(10)
    )
    assert manager.derive_seed("evaluation") != evolution_seed
    assert manager.derive_seed("evolution", 1) != evolution_seed


@pytest.mark.parametrize(
    ("namespace", "index"),
    [("", 0), ("evolution", -1), ("evolution", True)],
)
def test_invalid_named_seed_request_is_rejected(
    namespace: str,
    index: object,
) -> None:
    with pytest.raises(EvolutionConfigurationError):
        SeedManager(1).derive_seed(namespace, index)  # type: ignore[arg-type]


def test_metadata_capture_is_stable_when_record_fields_are_explicit() -> None:
    source = config()
    timestamp = "2026-08-29T12:00:00+00:00"

    first = ExperimentMetadata.capture(
        "bandit",
        42,
        source,
        tags=(("variant", "reference"), ("dataset", "toy")),
        created_at_utc=timestamp,
    )
    second = ExperimentMetadata.capture(
        "bandit",
        42,
        source,
        tags=(("dataset", "toy"), ("variant", "reference")),
        created_at_utc=timestamp,
    )

    assert first == second
    assert first.tags == (("dataset", "toy"), ("variant", "reference"))
    assert first.config_digest == source.digest
    assert first.master_seed == 42
    assert first.package_version
    assert first.python_version
    assert first.numpy_version
    assert first.platform


def test_metadata_requires_an_explicit_utc_timestamp() -> None:
    with pytest.raises(EvolutionConfigurationError, match="UTC offset"):
        ExperimentMetadata.capture(
            "bandit",
            42,
            config(),
            created_at_utc="2026-08-29T12:00:00+02:00",
        )
    with pytest.raises(EvolutionConfigurationError, match="created_at_utc"):
        ExperimentMetadata.capture(
            "bandit",
            42,
            config(),
            created_at_utc="",
        )
