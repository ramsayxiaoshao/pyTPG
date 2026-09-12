"""Evolution-specific configuration and invariant failures."""


class EvolutionError(Exception):
    """Base class for reference evolution failures."""


class EvolutionConfigurationError(EvolutionError):
    """Evolution configuration is invalid or internally inconsistent."""


class InvalidFitnessError(EvolutionError):
    """An evaluator returned a non-finite or non-real fitness."""


class MutationInvariantError(EvolutionError):
    """A mutation operator produced an invalid TPG graph."""


__all__ = [
    "EvolutionConfigurationError",
    "EvolutionError",
    "InvalidFitnessError",
    "MutationInvariantError",
]
