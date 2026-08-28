"""Low-level, environment-independent TPG domain model."""

from tpg.core.action import Action, AtomicAction, TeamReference
from tpg.core.graph import TPGGraph
from tpg.core.identifiers import (
    ActionID,
    InputIndex,
    LearnerID,
    OperatorName,
    ProgramID,
    RegisterIndex,
    TeamID,
)
from tpg.core.instruction import (
    ConstantOperand,
    InputOperand,
    Instruction,
    Operand,
    RegisterOperand,
)
from tpg.core.learner import Learner
from tpg.core.program import Program
from tpg.core.team import Team

__all__ = [
    "Action",
    "ActionID",
    "AtomicAction",
    "ConstantOperand",
    "InputIndex",
    "InputOperand",
    "Instruction",
    "Learner",
    "LearnerID",
    "Operand",
    "OperatorName",
    "Program",
    "ProgramID",
    "RegisterIndex",
    "RegisterOperand",
    "TPGGraph",
    "Team",
    "TeamID",
    "TeamReference",
]
