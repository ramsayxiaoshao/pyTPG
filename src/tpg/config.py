"""Immutable experiment-level composition of focused evolution configs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from tpg.evolution.errors import EvolutionConfigurationError
from tpg.evolution.genome import GenomeConfig, GenomeFactory
from tpg.evolution.initialization import GraphInitializer, InitializationConfig
from tpg.evolution.mutation import MutationConfig, default_mutation
from tpg.evolution.reproduction import Reproducer, ReproductionConfig
from tpg.evolution.selection import TournamentSelection
from tpg.runtime import OperatorRegistry


@dataclass(frozen=True, slots=True)
class TPGConfig:
    """All stable choices needed to rebuild the reference evolutionary system."""

    initialization: InitializationConfig
    mutation: MutationConfig = field(default_factory=MutationConfig)
    tournament_size: int = 3
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.initialization, InitializationConfig):
            raise EvolutionConfigurationError(
                "initialization must be an InitializationConfig"
            )
        if not isinstance(self.mutation, MutationConfig):
            raise EvolutionConfigurationError("mutation must be a MutationConfig")
        if not isinstance(self.reproduction, ReproductionConfig):
            raise EvolutionConfigurationError(
                "reproduction must be a ReproductionConfig"
            )
        if (
            isinstance(self.tournament_size, bool)
            or not isinstance(self.tournament_size, int)
            or self.tournament_size < 1
        ):
            raise EvolutionConfigurationError(
                "tournament_size must be a positive integer"
            )
        population_size = self.initialization.population_size
        if self.tournament_size > population_size:
            raise EvolutionConfigurationError(
                "tournament_size cannot exceed population_size"
            )
        if self.reproduction.elite_count > population_size:
            raise EvolutionConfigurationError(
                "elite_count cannot exceed population_size"
            )
        initial_team_size = self.initialization.learners_per_team
        if not (
            self.mutation.min_team_size
            <= initial_team_size
            <= self.mutation.max_team_size
        ):
            raise EvolutionConfigurationError(
                "learners_per_team must be within mutation team-size bounds"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible configuration representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> TPGConfig:
        """Rebuild a config from its exact version-1 field representation."""

        root = _mapping(value, "config")
        _require_keys(
            root,
            {"initialization", "mutation", "tournament_size", "reproduction"},
            "config",
        )
        initialization = _mapping(root["initialization"], "initialization")
        _require_keys(
            initialization,
            {
                "genome",
                "population_size",
                "team_count",
                "learners_per_team",
                "program_length",
            },
            "initialization",
        )
        genome = _mapping(initialization["genome"], "genome")
        mutation = _mapping(root["mutation"], "mutation")
        reproduction = _mapping(root["reproduction"], "reproduction")
        _require_keys(
            genome,
            {
                "input_size",
                "register_count",
                "n_actions",
                "min_program_length",
                "max_program_length",
                "constant_min",
                "constant_max",
            },
            "genome",
        )
        _require_keys(
            mutation,
            {
                "min_team_size",
                "max_team_size",
                "new_team_size",
                "instruction_insert_weight",
                "instruction_delete_weight",
                "instruction_modify_weight",
                "learner_action_weight",
                "learner_add_weight",
                "learner_delete_weight",
                "team_add_weight",
                "team_delete_weight",
            },
            "mutation",
        )
        _require_keys(
            reproduction,
            {"elite_count", "mutation_steps"},
            "reproduction",
        )
        try:
            return cls(
                initialization=InitializationConfig(
                    genome=GenomeConfig(**genome),
                    population_size=initialization["population_size"],
                    team_count=initialization["team_count"],
                    learners_per_team=initialization["learners_per_team"],
                    program_length=initialization["program_length"],
                ),
                mutation=MutationConfig(**mutation),
                tournament_size=root["tournament_size"],
                reproduction=ReproductionConfig(**reproduction),
            )
        except TypeError as error:
            raise EvolutionConfigurationError(
                f"configuration fields are malformed: {error}"
            ) from error

    @property
    def digest(self) -> str:
        """Return a SHA-256 digest of the canonical configuration JSON."""

        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def create_initializer(
        self,
        operators: OperatorRegistry | None = None,
    ) -> GraphInitializer:
        """Build the configured initializer and genome factory."""

        from tpg.evolution.initialization import GraphInitializer

        return GraphInitializer(self.initialization, operators)

    def create_reproducer(self, factory: GenomeFactory) -> Reproducer:
        """Build configured selection, mutation, and reproduction components."""

        return Reproducer(
            TournamentSelection(self.tournament_size),
            default_mutation(factory, self.mutation),
            self.reproduction,
        )


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvolutionConfigurationError(f"{location} must be an object")
    untyped = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in untyped):
        raise EvolutionConfigurationError(f"{location} must be an object")
    return cast(dict[str, Any], untyped)


def _require_keys(
    value: dict[str, Any],
    expected: set[str],
    location: str,
) -> None:
    if set(value) != expected:
        raise EvolutionConfigurationError(
            f"{location} fields must be exactly {sorted(expected)}"
        )

__all__ = ["TPGConfig"]
