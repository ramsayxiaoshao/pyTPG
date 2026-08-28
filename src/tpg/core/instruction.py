"""Instruction and operand value objects.

Execution and operator lookup intentionally belong to the future runtime layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from tpg.core._validation import require_non_negative_integer
from tpg.core.identifiers import InputIndex, OperatorName, RegisterIndex


@dataclass(frozen=True, slots=True)
class InputOperand:
    """Read one element of the normalized observation vector."""

    index: InputIndex

    @property
    def kind(self) -> Literal["input"]:
        """Return the stable operand discriminator used by the runtime."""

        return "input"

    def __post_init__(self) -> None:
        require_non_negative_integer(self.index, "input index")


@dataclass(frozen=True, slots=True)
class RegisterOperand:
    """Read one runtime register."""

    index: RegisterIndex

    @property
    def kind(self) -> Literal["register"]:
        """Return the stable operand discriminator used by the runtime."""

        return "register"

    def __post_init__(self) -> None:
        require_non_negative_integer(self.index, "register index")


@dataclass(frozen=True, slots=True)
class ConstantOperand:
    """Supply one finite constant value to an operator."""

    value: float

    @property
    def kind(self) -> Literal["constant"]:
        """Return the stable operand discriminator used by the runtime."""

        return "constant"

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            msg = f"constant must be a real number, got {self.value!r}"
            raise ValueError(msg)
        if not math.isfinite(self.value):
            msg = f"constant must be finite, got {self.value!r}"
            raise ValueError(msg)


Operand: TypeAlias = InputOperand | RegisterOperand | ConstantOperand


@dataclass(frozen=True, slots=True)
class Instruction:
    """One operator application with an explicit destination and operands."""

    operator: OperatorName
    destination: RegisterIndex
    operands: tuple[Operand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operator, str) or not self.operator.strip():
            raise ValueError("operator name must be a non-empty string")
        require_non_negative_integer(self.destination, "destination register index")
        if not isinstance(self.operands, tuple):
            raise TypeError("instruction operands must be a tuple")
        if not all(
            isinstance(item, (InputOperand, RegisterOperand, ConstantOperand))
            for item in self.operands
        ):
            raise TypeError("instruction operands contain an unsupported value")


__all__ = [
    "ConstantOperand",
    "InputOperand",
    "Instruction",
    "Operand",
    "RegisterOperand",
]
