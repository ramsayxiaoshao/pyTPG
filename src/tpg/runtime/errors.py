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
    """A winning team reference cannot be resolved by the Milestone 1 runtime."""


__all__ = [
    "InvalidInstructionError",
    "InvalidObservationError",
    "OperatorArityError",
    "OperatorExecutionError",
    "RegisterAccessError",
    "RuntimeConfigurationError",
    "TPGExecutionError",
    "TeamReferenceRequiresGraphError",
    "UnknownOperatorError",
]
