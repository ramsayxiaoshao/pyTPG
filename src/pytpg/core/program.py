"""Immutable program representation."""

from dataclasses import dataclass

from pytpg.core._validation import require_non_negative_integer
from pytpg.core.identifiers import ProgramID
from pytpg.core.instruction import Instruction


@dataclass(frozen=True, slots=True)
class Program:
    """A stable ID and non-empty ordered sequence of instructions."""

    id: ProgramID
    instructions: tuple[Instruction, ...]

    def __post_init__(self) -> None:
        require_non_negative_integer(self.id, "program ID")
        if not isinstance(self.instructions, tuple):
            raise TypeError("program instructions must be a tuple")
        if not self.instructions:
            raise ValueError("program must contain at least one instruction")
        if not all(isinstance(item, Instruction) for item in self.instructions):
            raise TypeError("program instructions contain an unsupported value")


__all__ = ["Program"]
