"""Deterministic reference runtime for programs and individual teams."""

from tpg.runtime.config import RuntimeConfig
from tpg.runtime.errors import (
    InvalidInstructionError,
    InvalidObservationError,
    OperatorArityError,
    OperatorExecutionError,
    RegisterAccessError,
    RuntimeConfigurationError,
    TeamReferenceRequiresGraphError,
    TPGExecutionError,
    UnknownOperatorError,
)
from tpg.runtime.executor import ExecutionResult, ProgramExecutor
from tpg.runtime.operators import (
    Operator,
    OperatorFunction,
    OperatorRegistry,
    default_operator_registry,
)
from tpg.runtime.registers import RegisterFile
from tpg.runtime.team_runtime import DeterministicRuntime

__all__ = [
    "DeterministicRuntime",
    "ExecutionResult",
    "InvalidInstructionError",
    "InvalidObservationError",
    "Operator",
    "OperatorArityError",
    "OperatorExecutionError",
    "OperatorFunction",
    "OperatorRegistry",
    "ProgramExecutor",
    "RegisterAccessError",
    "RegisterFile",
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "TPGExecutionError",
    "TeamReferenceRequiresGraphError",
    "UnknownOperatorError",
    "default_operator_registry",
]
