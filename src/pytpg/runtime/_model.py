"""Structural model contracts that keep runtime below the core package."""

from typing import Literal, Protocol, TypeAlias


class InputOperandLike(Protocol):
    """Read-only shape required for an input operand."""

    @property
    def kind(self) -> Literal["input"]: ...

    @property
    def index(self) -> int: ...


class RegisterOperandLike(Protocol):
    """Read-only shape required for a register operand."""

    @property
    def kind(self) -> Literal["register"]: ...

    @property
    def index(self) -> int: ...


class ConstantOperandLike(Protocol):
    """Read-only shape required for a constant operand."""

    @property
    def kind(self) -> Literal["constant"]: ...

    @property
    def value(self) -> float: ...


OperandLike: TypeAlias = InputOperandLike | RegisterOperandLike | ConstantOperandLike


class InstructionLike(Protocol):
    """Read-only instruction shape consumed by the executor."""

    @property
    def operator(self) -> str: ...

    @property
    def destination(self) -> int: ...

    @property
    def operands(self) -> tuple[OperandLike, ...]: ...


class ProgramLike(Protocol):
    """Read-only program shape consumed by the executor."""

    @property
    def id(self) -> int: ...

    @property
    def instructions(self) -> tuple[InstructionLike, ...]: ...


class AtomicActionLike(Protocol):
    """Read-only atomic action shape consumed during team selection."""

    @property
    def kind(self) -> Literal["atomic"]: ...

    @property
    def action_id(self) -> int: ...


class TeamReferenceLike(Protocol):
    """Read-only team-reference shape recognized but not traversed yet."""

    @property
    def kind(self) -> Literal["team_reference"]: ...

    @property
    def team_id(self) -> int: ...


ActionLike: TypeAlias = AtomicActionLike | TeamReferenceLike


class LearnerLike(Protocol):
    """Read-only learner shape consumed by team selection."""

    @property
    def id(self) -> int: ...

    @property
    def program(self) -> ProgramLike: ...

    @property
    def action(self) -> ActionLike: ...


class TeamLike(Protocol):
    """Read-only team shape consumed by the deterministic runtime."""

    @property
    def id(self) -> int: ...

    @property
    def learners(self) -> tuple[LearnerLike, ...]: ...


class GraphLike(Protocol):
    """Read-only graph shape consumed by validation and traversal."""

    @property
    def teams(self) -> tuple[TeamLike, ...]: ...

    @property
    def root_team_ids(self) -> tuple[int, ...]: ...
