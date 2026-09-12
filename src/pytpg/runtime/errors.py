"""Exceptions raised by the deterministic reference runtime."""


class TPGExecutionError(Exception):
    """Base class for deterministic runtime failures."""


class RuntimeConfigurationError(TPGExecutionError):
    """The runtime configuration is internally invalid."""


class InvalidObservationError(TPGExecutionError):
    """An observation does not satisfy the configured input contract."""


class RegisterAccessError(TPGExecutionError):
    """A register index or register value is invalid."""


class InvalidInstructionError(TPGExecutionError):
    """An instruction cannot execute under the current runtime configuration."""


class UnknownOperatorError(InvalidInstructionError):
    """An instruction names an operator absent from its registry."""


class OperatorArityError(InvalidInstructionError):
    """An instruction supplies the wrong number of operands."""


class OperatorExecutionError(TPGExecutionError):
    """An operator violates the runtime operator contract."""


class TeamReferenceRequiresGraphError(TPGExecutionError):
    """A single-team runtime cannot resolve a winning team reference."""


class RootSelectionError(TPGExecutionError):
    """A traversal root is missing, ambiguous, or not declared by the graph."""


class MissingTeamError(TPGExecutionError):
    """Traversal encounters a team ID absent from the graph."""


class NoEligibleLearnerError(TPGExecutionError):
    """Visited-edge exclusion leaves a team without an eligible learner."""


class TraversalLimitExceededError(TPGExecutionError):
    """Graph traversal reaches its configured deterministic step limit."""


__all__ = [
    "InvalidInstructionError",
    "InvalidObservationError",
    "MissingTeamError",
    "NoEligibleLearnerError",
    "OperatorArityError",
    "OperatorExecutionError",
    "RegisterAccessError",
    "RootSelectionError",
    "RuntimeConfigurationError",
    "TPGExecutionError",
    "TeamReferenceRequiresGraphError",
    "TraversalLimitExceededError",
    "UnknownOperatorError",
]
