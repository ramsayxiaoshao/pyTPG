"""Statically distinct identifiers used by the core model."""

from typing import NewType

ActionID = NewType("ActionID", int)
InputIndex = NewType("InputIndex", int)
LearnerID = NewType("LearnerID", int)
ProgramID = NewType("ProgramID", int)
RegisterIndex = NewType("RegisterIndex", int)
TeamID = NewType("TeamID", int)

OperatorName = NewType("OperatorName", str)

__all__ = [
    "ActionID",
    "InputIndex",
    "LearnerID",
    "OperatorName",
    "ProgramID",
    "RegisterIndex",
    "TeamID",
]
