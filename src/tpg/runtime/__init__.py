"""Deterministic reference runtime for programs and individual teams."""

from tpg.runtime.config import RuntimeConfig
from tpg.runtime.errors import (
    InvalidInstructionError,
    InvalidObservationError,
    MissingTeamError,
    NoEligibleLearnerError,
    OperatorArityError,
    OperatorExecutionError,
    RegisterAccessError,
    RootSelectionError,
    RuntimeConfigurationError,
    TeamReferenceRequiresGraphError,
    TPGExecutionError,
    TraversalLimitExceededError,
    UnknownOperatorError,
)
from tpg.runtime.executor import ExecutionResult, ProgramExecutor
from tpg.runtime.graph_runtime import (
    GraphRuntime,
    TraversalActionKind,
    TraversalResult,
    TraversalStep,
)
from tpg.runtime.graph_validation import (
    GraphValidationError,
    GraphValidationIssue,
    GraphValidationReport,
    GraphValidator,
    ValidationSeverity,
)
from tpg.runtime.inspection import (
    GraphSummary,
    cyclic_team_ids,
    reachable_team_ids,
    summarize_graph,
)
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
    "GraphRuntime",
    "GraphSummary",
    "GraphValidationError",
    "GraphValidationIssue",
    "GraphValidationReport",
    "GraphValidator",
    "InvalidInstructionError",
    "InvalidObservationError",
    "MissingTeamError",
    "NoEligibleLearnerError",
    "Operator",
    "OperatorArityError",
    "OperatorExecutionError",
    "OperatorFunction",
    "OperatorRegistry",
    "ProgramExecutor",
    "RegisterAccessError",
    "RegisterFile",
    "RootSelectionError",
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "TPGExecutionError",
    "TeamReferenceRequiresGraphError",
    "TraversalActionKind",
    "TraversalLimitExceededError",
    "TraversalResult",
    "TraversalStep",
    "UnknownOperatorError",
    "ValidationSeverity",
    "cyclic_team_ids",
    "default_operator_registry",
    "reachable_team_ids",
    "summarize_graph",
]
