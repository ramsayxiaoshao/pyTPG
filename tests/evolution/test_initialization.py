"""Initialization and RNG reproducibility contracts."""

import pytest

from tpg.evolution import (
    EvolutionConfigurationError,
    GenomeConfig,
    GraphInitializer,
    InitializationConfig,
    PopulationInitializer,
    create_rng,
)
from tpg.runtime import reachable_team_ids


def initialization_config(**overrides: object) -> InitializationConfig:
    values: dict[str, object] = {
        "genome": GenomeConfig(2, 3, 4, max_program_length=8),
        "population_size": 4,
        "team_count": 3,
        "learners_per_team": 3,
        "program_length": 2,
    }
    values.update(overrides)
    return InitializationConfig(**values)  # type: ignore[arg-type]


def test_same_seed_creates_equal_populations() -> None:
    initializer = PopulationInitializer(GraphInitializer(initialization_config()))

    first = initializer.initialize(create_rng(1234))
    second = initializer.initialize(create_rng(1234))

    assert first == second
    assert first.generation == 0
    assert len(first) == 4


def test_rng_is_an_explicit_numpy_generator_with_reproducible_stream() -> None:
    first = create_rng(77)
    second = create_rng(77)

    assert type(first).__name__ == "Generator"
    assert tuple(int(first.integers(1_000_000)) for _ in range(8)) == tuple(
        int(second.integers(1_000_000)) for _ in range(8)
    )


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7"])
def test_invalid_seed_is_rejected(seed: object) -> None:
    with pytest.raises(EvolutionConfigurationError, match="seed"):
        create_rng(seed)  # type: ignore[arg-type]


def test_multi_team_initializer_builds_one_reachable_forward_chain() -> None:
    initializer = GraphInitializer(initialization_config(population_size=1))

    graph = initializer.initialize(create_rng(19))

    assert graph.root_team_ids == (0,)
    assert reachable_team_ids(graph) == {0, 1, 2}
    assert graph.validate(
        initializer.factory.runtime_config,
        operators=initializer.factory.operators,
    ).issues == ()
    assert tuple(len(team.learners) for team in graph.teams) == (3, 3, 3)
    assert tuple(
        team.learners[0].action.kind for team in graph.teams
    ) == ("team_reference", "team_reference", "atomic")


def test_initializer_assigns_graph_local_unique_ids_and_bid_writers() -> None:
    initializer = GraphInitializer(initialization_config(population_size=1))
    graph = initializer.initialize(create_rng(21))
    learners = tuple(learner for team in graph.teams for learner in team.learners)

    assert len({learner.id for learner in learners}) == len(learners)
    assert len({learner.program.id for learner in learners}) == len(learners)
    assert all(
        learner.program.instructions[-1].destination == 0 for learner in learners
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"population_size": 0}, "population_size"),
        ({"team_count": 0}, "team_count"),
        ({"learners_per_team": 1}, "at least 2"),
        ({"program_length": 9}, "program_length"),
    ],
)
def test_invalid_initialization_config_is_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(EvolutionConfigurationError, match=message):
        initialization_config(**overrides)


@pytest.mark.parametrize(
    "bounds",
    [(float("nan"), 1.0), (-1.0, float("inf")), (True, 1.0)],
)
def test_genome_rejects_nonfinite_or_boolean_constant_bounds(
    bounds: tuple[object, object],
) -> None:
    with pytest.raises(EvolutionConfigurationError, match="finite real"):
        GenomeConfig(1, 1, 2, constant_min=bounds[0], constant_max=bounds[1])  # type: ignore[arg-type]
