"""Extensible operators and protected built-in arithmetic."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeAlias

from tpg.runtime.errors import UnknownOperatorError

OperatorFunction: TypeAlias = Callable[[tuple[float, ...]], float]


@dataclass(frozen=True, slots=True)
class Operator:
    """A named fixed-arity operation callable by the program executor."""

    name: str
    arity: int
    function: OperatorFunction

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("operator name must be a non-empty string")
        if isinstance(self.arity, bool) or not isinstance(self.arity, int):
            raise ValueError("operator arity must be an integer")
        if self.arity < 1:
            raise ValueError("operator arity must be positive")
        if not callable(self.function):
            raise TypeError("operator function must be callable")


class OperatorRegistry:
    """An explicitly owned registry with no process-global mutable state."""

    __slots__ = ("_operators",)

    def __init__(self, operators: Iterable[Operator] = ()) -> None:
        self._operators: dict[str, Operator] = {}
        for operator in operators:
            self.register(operator)

    def register(self, operator: Operator) -> None:
        """Register an operator, rejecting ambiguous replacement."""

        if not isinstance(operator, Operator):
            raise TypeError("registry accepts Operator values")
        if operator.name in self._operators:
            msg = f"operator {operator.name!r} is already registered"
            raise ValueError(msg)
        self._operators[operator.name] = operator

    def resolve(self, name: str) -> Operator:
        """Resolve a name or raise a runtime-specific diagnostic."""

        try:
            return self._operators[name]
        except KeyError as error:
            msg = f"unknown operator {name!r}"
            raise UnknownOperatorError(msg) from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return names in deterministic registration order."""

        return tuple(self._operators)


def _identity(values: tuple[float, ...]) -> float:
    return values[0]


def _add(values: tuple[float, ...]) -> float:
    return values[0] + values[1]


def _subtract(values: tuple[float, ...]) -> float:
    return values[0] - values[1]


def _multiply(values: tuple[float, ...]) -> float:
    return values[0] * values[1]


def _divide(values: tuple[float, ...]) -> float:
    return 0.0 if values[1] == 0.0 else values[0] / values[1]


def _sin(values: tuple[float, ...]) -> float:
    return math.sin(values[0])


def _cos(values: tuple[float, ...]) -> float:
    return math.cos(values[0])


def _log(values: tuple[float, ...]) -> float:
    return 0.0 if values[0] <= 0.0 else math.log(values[0])


def _minimum(values: tuple[float, ...]) -> float:
    return min(values[0], values[1])


def _maximum(values: tuple[float, ...]) -> float:
    return max(values[0], values[1])


def default_operator_registry() -> OperatorRegistry:
    """Create a fresh registry containing the deterministic reference set."""

    definitions: tuple[tuple[str, int, OperatorFunction], ...] = (
        ("identity", 1, _identity),
        ("add", 2, _add),
        ("subtract", 2, _subtract),
        ("multiply", 2, _multiply),
        ("divide", 2, _divide),
        ("sin", 1, _sin),
        ("cos", 1, _cos),
        ("log", 1, _log),
        ("min", 2, _minimum),
        ("max", 2, _maximum),
    )
    return OperatorRegistry(
        Operator(name, arity, function) for name, arity, function in definitions
    )


__all__ = [
    "Operator",
    "OperatorFunction",
    "OperatorRegistry",
    "default_operator_registry",
]
