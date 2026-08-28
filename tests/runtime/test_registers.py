"""Unit tests for deterministic register storage."""

import pytest

from tpg.core import RegisterIndex
from tpg.runtime import RegisterAccessError, RegisterFile


def test_registers_start_at_zero_and_snapshot_is_immutable() -> None:
    registers = RegisterFile(3)

    assert registers.count == 3
    assert registers.snapshot() == (0.0, 0.0, 0.0)


def test_register_write_normalizes_real_values_to_float() -> None:
    registers = RegisterFile(2)

    registers.write(RegisterIndex(1), 7)

    assert registers.read(RegisterIndex(1)) == 7.0
    assert isinstance(registers.read(RegisterIndex(1)), float)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_register_file_rejects_invalid_counts(count: object) -> None:
    with pytest.raises(RegisterAccessError, match="positive integer"):
        RegisterFile(count)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [-1, 2])
def test_register_access_rejects_out_of_bounds_indices(index: int) -> None:
    registers = RegisterFile(2)

    with pytest.raises(RegisterAccessError, match="outside"):
        registers.read(RegisterIndex(index))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_register_write_rejects_non_finite_or_boolean_values(value: float) -> None:
    registers = RegisterFile(1)

    with pytest.raises(RegisterAccessError, match="real number|finite"):
        registers.write(RegisterIndex(0), value)
