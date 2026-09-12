"""Learner value object."""

from dataclasses import dataclass

from pytpg.core._validation import require_non_negative_integer
from pytpg.core.action import Action, AtomicAction, TeamReference
from pytpg.core.identifiers import LearnerID
from pytpg.core.program import Program


@dataclass(frozen=True, slots=True)
class Learner:
    """The immutable composition of a bidding program and an action."""

    id: LearnerID
    program: Program
    action: Action

    def __post_init__(self) -> None:
        require_non_negative_integer(self.id, "learner ID")
        if not isinstance(self.program, Program):
            raise TypeError("learner program must be a Program")
        if not isinstance(self.action, (AtomicAction, TeamReference)):
            raise TypeError("learner action must be an AtomicAction or TeamReference")


__all__ = ["Learner"]
