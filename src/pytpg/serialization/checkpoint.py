"""Version-1 JSON checkpoints at evaluated generation boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from pytpg.config import TPGConfig
from pytpg.evaluation.statistics import (
    GenerationStatistics,
    OperatorStatistics,
    RunStatistics,
)
from pytpg.evolution import (
    EvaluatedIndividual,
    EvaluatedPopulation,
    EvolutionRunResult,
)
from pytpg.evolution.errors import EvolutionConfigurationError
from pytpg.evolution.rng import RandomGenerator
from pytpg.metadata import ExperimentMetadata
from pytpg.serialization._validation import (
    exact_keys,
    finite_number,
    integer,
    mapping,
    sequence,
    string,
)
from pytpg.serialization.errors import (
    CheckpointCompatibilityError,
    InvalidSerializedDataError,
    UnsupportedFormatVersionError,
)
from pytpg.serialization.graph import graph_from_data, graph_to_data
from pytpg.serialization.io import dump_json, load_json, read_text, write_atomic
from pytpg.serialization.rng import RNGSnapshot, capture_rng, restore_rng

CHECKPOINT_FORMAT = "pytpg-checkpoint"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Sufficient state to continue a deterministic reference evolution."""

    config: TPGConfig
    metadata: ExperimentMetadata
    evaluated: EvaluatedPopulation
    rng: RNGSnapshot
    statistics: RunStatistics

    def __post_init__(self) -> None:
        if self.metadata.config_digest != self.config.digest:
            raise CheckpointCompatibilityError(
                "metadata config digest does not match checkpoint config"
            )
        expected_size = self.config.initialization.population_size
        if len(self.evaluated) != expected_size:
            raise CheckpointCompatibilityError(
                "checkpoint population size does not match configuration"
            )
        if self.statistics.generations[-1].generation != self.evaluated.generation:
            raise CheckpointCompatibilityError(
                "checkpoint statistics do not end at evaluated generation"
            )
        expected_final_statistics = GenerationStatistics.calculate(self.evaluated)
        if self.statistics.generations[-1] != expected_final_statistics:
            raise CheckpointCompatibilityError(
                "checkpoint final statistics do not match evaluated population"
            )
        restore_rng(self.rng)
        initializer = self.config.create_initializer()
        for position, individual in enumerate(self.evaluated.individuals):
            report = individual.graph.validate(
                initializer.factory.runtime_config,
                operators=initializer.factory.operators,
            )
            if not report.is_valid:
                codes = ", ".join(issue.code for issue in report.errors)
                raise CheckpointCompatibilityError(
                    f"checkpoint graph {position} is invalid: {codes}"
                )

    @classmethod
    def from_run(
        cls,
        config: TPGConfig,
        metadata: ExperimentMetadata,
        result: EvolutionRunResult,
        rng: RandomGenerator,
        *,
        statistics: RunStatistics | None = None,
    ) -> Checkpoint:
        """Capture the final evaluated boundary and next RNG state."""

        return cls(
            config=config,
            metadata=metadata,
            evaluated=result.evaluations[-1],
            rng=capture_rng(rng),
            statistics=statistics or RunStatistics.from_run(result),
        )


def checkpoint_to_dict(checkpoint: Checkpoint) -> dict[str, object]:
    """Return the complete versioned checkpoint document."""

    return {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint": {
            "config": checkpoint.config.to_dict(),
            "metadata": _metadata_to_data(checkpoint.metadata),
            "evaluated_population": _evaluated_to_data(checkpoint.evaluated),
            "rng": checkpoint.rng.to_dict(),
            "statistics": _statistics_to_data(checkpoint.statistics),
        },
    }


def checkpoint_from_dict(value: object) -> Checkpoint:
    """Strictly parse and validate a supported checkpoint document."""

    document = mapping(value, "document")
    exact_keys(document, {"format", "format_version", "checkpoint"}, "document")
    if string(document["format"], "document.format") != CHECKPOINT_FORMAT:
        raise InvalidSerializedDataError("document is not a pyTPG checkpoint")
    version = integer(document["format_version"], "document.format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise UnsupportedFormatVersionError(
            f"unsupported checkpoint format version {version}"
        )
    data = mapping(document["checkpoint"], "checkpoint")
    exact_keys(
        data,
        {"config", "metadata", "evaluated_population", "rng", "statistics"},
        "checkpoint",
    )
    try:
        return Checkpoint(
            config=TPGConfig.from_dict(data["config"]),
            metadata=_metadata_from_data(data["metadata"]),
            evaluated=_evaluated_from_data(data["evaluated_population"]),
            rng=RNGSnapshot.from_dict(data["rng"]),
            statistics=_statistics_from_data(data["statistics"]),
        )
    except EvolutionConfigurationError as error:
        raise InvalidSerializedDataError(
            f"invalid checkpoint component: {error}"
        ) from error


def dumps_checkpoint(checkpoint: Checkpoint) -> str:
    return dump_json(checkpoint_to_dict(checkpoint))


def loads_checkpoint(text: str) -> Checkpoint:
    return checkpoint_from_dict(load_json(text))


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> None:
    write_atomic(path, dumps_checkpoint(checkpoint))


def load_checkpoint(path: str | Path) -> Checkpoint:
    return loads_checkpoint(read_text(path))


def _metadata_to_data(metadata: ExperimentMetadata) -> dict[str, object]:
    data = asdict(metadata)
    data["tags"] = [list(item) for item in metadata.tags]
    return data


def _metadata_from_data(value: object) -> ExperimentMetadata:
    item = mapping(value, "checkpoint.metadata")
    expected = {
        "name",
        "run_id",
        "created_at_utc",
        "master_seed",
        "config_digest",
        "package_version",
        "python_version",
        "numpy_version",
        "platform",
        "tags",
    }
    exact_keys(item, expected, "checkpoint.metadata")
    tags = tuple(
        _tag_from_data(tag, f"checkpoint.metadata.tags[{index}]")
        for index, tag in enumerate(
            sequence(item["tags"], "checkpoint.metadata.tags")
        )
    )
    return ExperimentMetadata(
        name=string(item["name"], "checkpoint.metadata.name"),
        run_id=string(item["run_id"], "checkpoint.metadata.run_id"),
        created_at_utc=string(
            item["created_at_utc"], "checkpoint.metadata.created_at_utc"
        ),
        master_seed=integer(
            item["master_seed"], "checkpoint.metadata.master_seed"
        ),
        config_digest=string(
            item["config_digest"], "checkpoint.metadata.config_digest"
        ),
        package_version=string(
            item["package_version"], "checkpoint.metadata.package_version"
        ),
        python_version=string(
            item["python_version"], "checkpoint.metadata.python_version"
        ),
        numpy_version=string(
            item["numpy_version"], "checkpoint.metadata.numpy_version"
        ),
        platform=string(item["platform"], "checkpoint.metadata.platform"),
        tags=tags,
    )


def _tag_from_data(value: object, location: str) -> tuple[str, str]:
    values = sequence(value, location)
    if len(values) != 2:
        raise InvalidSerializedDataError(f"{location} must contain two strings")
    return (string(values[0], f"{location}[0]"), string(values[1], f"{location}[1]"))


def _evaluated_to_data(evaluated: EvaluatedPopulation) -> dict[str, object]:
    return {
        "generation": evaluated.generation,
        "individuals": [
            {
                "position": individual.position,
                "fitness": individual.fitness,
                "graph": graph_to_data(individual.graph),
            }
            for individual in evaluated.individuals
        ],
    }


def _evaluated_from_data(value: object) -> EvaluatedPopulation:
    data = mapping(value, "checkpoint.evaluated_population")
    exact_keys(data, {"generation", "individuals"}, "checkpoint.evaluated_population")
    generation = integer(
        data["generation"], "checkpoint.evaluated_population.generation"
    )
    individuals = tuple(
        _evaluated_individual_from_data(
            raw,
            f"checkpoint.evaluated_population.individuals[{index}]",
        )
        for index, raw in enumerate(
            sequence(
                data["individuals"],
                "checkpoint.evaluated_population.individuals",
            )
        )
    )
    return EvaluatedPopulation(generation, individuals)


def _evaluated_individual_from_data(
    value: object,
    location: str,
) -> EvaluatedIndividual:
    item = mapping(value, location)
    exact_keys(item, {"position", "fitness", "graph"}, location)
    return EvaluatedIndividual(
        position=integer(item["position"], f"{location}.position"),
        graph=graph_from_data(item["graph"]),
        fitness=finite_number(item["fitness"], f"{location}.fitness"),
    )


def _statistics_to_data(statistics: RunStatistics) -> list[dict[str, object]]:
    return [
        {
            "generation": item.generation,
            "population_size": item.population_size,
            "minimum_fitness": item.minimum_fitness,
            "maximum_fitness": item.maximum_fitness,
            "mean_fitness": item.mean_fitness,
            "median_fitness": item.median_fitness,
            "population_stddev": item.population_stddev,
            "best_position": item.best_position,
            "elite_count": item.elite_count,
            "mutation_attempts": item.mutation_attempts,
            "applied_mutations": item.applied_mutations,
            "operators": [asdict(operator) for operator in item.operators],
        }
        for item in statistics.generations
    ]


def _statistics_from_data(value: object) -> RunStatistics:
    generations = tuple(
        _generation_statistics_from_data(
            item, f"checkpoint.statistics[{index}]"
        )
        for index, item in enumerate(sequence(value, "checkpoint.statistics"))
    )
    return RunStatistics(generations)


def _generation_statistics_from_data(
    value: object,
    location: str,
) -> GenerationStatistics:
    item = mapping(value, location)
    expected = {
        "generation",
        "population_size",
        "minimum_fitness",
        "maximum_fitness",
        "mean_fitness",
        "median_fitness",
        "population_stddev",
        "best_position",
        "elite_count",
        "mutation_attempts",
        "applied_mutations",
        "operators",
    }
    exact_keys(item, expected, location)
    return GenerationStatistics(
        generation=integer(item["generation"], f"{location}.generation"),
        population_size=integer(
            item["population_size"], f"{location}.population_size", minimum=1
        ),
        minimum_fitness=finite_number(
            item["minimum_fitness"], f"{location}.minimum_fitness"
        ),
        maximum_fitness=finite_number(
            item["maximum_fitness"], f"{location}.maximum_fitness"
        ),
        mean_fitness=finite_number(
            item["mean_fitness"], f"{location}.mean_fitness"
        ),
        median_fitness=finite_number(
            item["median_fitness"], f"{location}.median_fitness"
        ),
        population_stddev=finite_number(
            item["population_stddev"], f"{location}.population_stddev"
        ),
        best_position=integer(
            item["best_position"], f"{location}.best_position"
        ),
        elite_count=integer(item["elite_count"], f"{location}.elite_count"),
        mutation_attempts=integer(
            item["mutation_attempts"], f"{location}.mutation_attempts"
        ),
        applied_mutations=integer(
            item["applied_mutations"], f"{location}.applied_mutations"
        ),
        operators=tuple(
            _operator_statistics_from_data(
                operator, f"{location}.operators[{index}]"
            )
            for index, operator in enumerate(
                sequence(item["operators"], f"{location}.operators")
            )
        ),
    )


def _operator_statistics_from_data(
    value: object,
    location: str,
) -> OperatorStatistics:
    item = mapping(value, location)
    exact_keys(item, {"operator", "attempts", "applied"}, location)
    return OperatorStatistics(
        operator=string(item["operator"], f"{location}.operator"),
        attempts=integer(item["attempts"], f"{location}.attempts", minimum=1),
        applied=integer(item["applied"], f"{location}.applied"),
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "Checkpoint",
    "checkpoint_from_dict",
    "checkpoint_to_dict",
    "dumps_checkpoint",
    "load_checkpoint",
    "loads_checkpoint",
    "save_checkpoint",
]
