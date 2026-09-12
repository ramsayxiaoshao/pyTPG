"""Optional environment adapters kept outside the TPG core."""

from pytpg.adapters.errors import (
    GymnasiumDependencyError,
    GymnasiumEpisodeStateError,
    GymnasiumEvaluationError,
    GymnasiumIntegrationError,
    InvalidGymnasiumActionError,
    InvalidGymnasiumTransitionError,
    UnsupportedGymnasiumSpaceError,
)
from pytpg.adapters.gymnasium import (
    GymnasiumAdapter,
    GymnasiumEnvironment,
    GymnasiumEnvironmentFactory,
    GymnasiumReset,
    GymnasiumStep,
    make_gymnasium_environment,
)
from pytpg.adapters.gymnasium_evaluation import (
    GymnasiumEpisodeResult,
    GymnasiumEvaluationConfig,
    GymnasiumFitness,
)

__all__ = [
    "GymnasiumAdapter",
    "GymnasiumDependencyError",
    "GymnasiumEnvironment",
    "GymnasiumEnvironmentFactory",
    "GymnasiumEpisodeResult",
    "GymnasiumEpisodeStateError",
    "GymnasiumEvaluationConfig",
    "GymnasiumEvaluationError",
    "GymnasiumFitness",
    "GymnasiumIntegrationError",
    "GymnasiumReset",
    "GymnasiumStep",
    "InvalidGymnasiumActionError",
    "InvalidGymnasiumTransitionError",
    "UnsupportedGymnasiumSpaceError",
    "make_gymnasium_environment",
]
