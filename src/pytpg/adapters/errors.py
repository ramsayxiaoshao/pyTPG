"""Failures raised at the optional Gymnasium integration boundary."""


class GymnasiumIntegrationError(Exception):
    """Base class for Gymnasium adapter and evaluation failures."""


class GymnasiumDependencyError(GymnasiumIntegrationError):
    """The optional Gymnasium dependency is unavailable."""


class UnsupportedGymnasiumSpaceError(GymnasiumIntegrationError):
    """An environment space cannot be represented by the current adapter."""


class InvalidGymnasiumTransitionError(GymnasiumIntegrationError):
    """An environment returned a value outside the modern Gymnasium API."""


class GymnasiumEpisodeStateError(GymnasiumIntegrationError):
    """An operation conflicts with the adapter's episode lifecycle."""


class InvalidGymnasiumActionError(GymnasiumIntegrationError):
    """A TPG action ID cannot be mapped into the environment action space."""


class GymnasiumEvaluationError(GymnasiumIntegrationError):
    """A Gymnasium fitness-evaluation configuration or result is invalid."""


__all__ = [
    "GymnasiumDependencyError",
    "GymnasiumEpisodeStateError",
    "GymnasiumEvaluationError",
    "GymnasiumIntegrationError",
    "InvalidGymnasiumActionError",
    "InvalidGymnasiumTransitionError",
    "UnsupportedGymnasiumSpaceError",
]
