"""Unit tests for the extensible operator registry."""

import math

import pytest

from pytpg.core import OperatorName
from pytpg.runtime import (
    Operator,
    OperatorRegistry,
    UnknownOperatorError,
    default_operator_registry,
)


def test_default_registry_has_stable_documented_order() -> None:
    registry = default_operator_registry()

    assert registry.names == (
        "identity",
        "add",
        "subtract",
        "multiply",
        "divide",
        "sin",
        "cos",
        "log",
        "min",
        "max",
    )


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("identity", (2.5,), 2.5),
        ("add", (5.0, 2.0), 7.0),
        ("subtract", (5.0, 2.0), 3.0),
        ("multiply", (5.0, 2.0), 10.0),
        ("divide", (5.0, 2.0), 2.5),
        ("sin", (0.0,), 0.0),
        ("cos", (0.0,), 1.0),
        ("log", (math.e,), 1.0),
        ("min", (5.0, 2.0), 2.0),
        ("max", (5.0, 2.0), 5.0),
    ],
)
def test_default_operator_results(
    name: str,
    values: tuple[float, ...],
    expected: float,
) -> None:
    operator = default_operator_registry().resolve(OperatorName(name))

    assert math.isclose(operator.function(values), expected)


def test_protected_division_by_zero_returns_zero() -> None:
    operator = default_operator_registry().resolve(OperatorName("divide"))

    assert operator.function((9.0, 0.0)) == 0.0


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_protected_log_of_non_positive_value_returns_zero(value: float) -> None:
    operator = default_operator_registry().resolve(OperatorName("log"))

    assert operator.function((value,)) == 0.0


def test_registry_accepts_custom_operator_without_executor_changes() -> None:
    square = Operator(OperatorName("square"), 1, lambda values: values[0] ** 2)
    registry = OperatorRegistry((square,))

    assert registry.resolve(OperatorName("square")).function((3.0,)) == 9.0


def test_registry_rejects_duplicate_names() -> None:
    registry = OperatorRegistry()
    identity = Operator(OperatorName("identity"), 1, lambda values: values[0])
    registry.register(identity)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(identity)


def test_default_registries_do_not_share_mutable_state() -> None:
    first = default_operator_registry()
    second = default_operator_registry()
    first.register(Operator(OperatorName("square"), 1, lambda values: values[0] ** 2))

    with pytest.raises(UnknownOperatorError):
        second.resolve(OperatorName("square"))


def test_unknown_operator_has_specific_diagnostic() -> None:
    with pytest.raises(UnknownOperatorError, match="unknown operator"):
        OperatorRegistry().resolve(OperatorName("missing"))
