"""Multi-agent controller configuration and execution failures."""


class TPGMultiAgentError(Exception):
    """Base class for multi-agent composition failures."""


class MultiAgentConfigurationError(TPGMultiAgentError):
    """The agent roster, controller, graph, or action space is inconsistent."""


class MultiAgentStateError(TPGMultiAgentError):
    """A multi-agent episode lifecycle transition is invalid."""


class MultiAgentObservationError(TPGMultiAgentError):
    """A joint observation is missing, unexpected, malformed, or non-finite."""


class MultiAgentActionError(TPGMultiAgentError):
    """An individual or encoded joint action is outside its declared space."""


__all__ = [
    "MultiAgentActionError",
    "MultiAgentConfigurationError",
    "MultiAgentObservationError",
    "MultiAgentStateError",
    "TPGMultiAgentError",
]
