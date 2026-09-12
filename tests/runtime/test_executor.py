"""Unit and regression tests for deterministic program execution."""

import pytest

from pytpg.core import (
    ConstantOperand,
    InputIndex,
    InputOperand,
    Instruction,
    OperatorName,
    Program,
    ProgramID,
    RegisterIndex,
    RegisterOperand,
)
from pytpg.runtime import (
    InvalidInstructionError,
    InvalidObservationError,
    Operator,
    OperatorArityError,
    OperatorExecutionError,
    OperatorRegistry,
    ProgramExecutor,
    RuntimeConfig,
    RuntimeConfigurationError,
    UnknownOperatorError,
)


def make_program(*instructions: Instruction) -> Program:
    return Program(ProgramID(0), instructions)


def test_program_executes_in_order_and_returns_register_zero() -> None:
    program = make_program(
        Instruction(
            OperatorName("add"),
            RegisterIndex(1),
            (InputOperand(InputIndex(0)), ConstantOperand(2.0)),
        ),
        Instruction(
            OperatorName("multiply"),
            RegisterIndex(0),
            (RegisterOperand(RegisterIndex(1)), InputOperand(InputIndex(1))),
        ),
    )
    executor = ProgramExecutor(RuntimeConfig(input_size=2, register_count=3))

    result = executor.execute(program, (3.0, 4.0))

    assert result.output == 20.0
    assert result.registers == (20.0, 5.0, 0.0)


def test_each_execution_gets_fresh_zero_registers() -> None:
    program = make_program(
        Instruction(
            OperatorName("add"),
            RegisterIndex(0),
            (RegisterOperand(RegisterIndex(0)), ConstantOperand(1.0)),
        )
    )
    executor = ProgramExecutor(RuntimeConfig(input_size=0, register_count=1))

    first = executor.execute(program, ())
    second = executor.execute(program, ())

    assert first == second
    assert first.output == 1.0


def test_output_is_raw_and_may_be_negative() -> None:
    program = make_program(
        Instruction(
            OperatorName("subtract"),
            RegisterIndex(0),
            (ConstantOperand(1.0), ConstantOperand(4.0)),
        )
    )

    result = ProgramExecutor(RuntimeConfig(0, 1)).execute(program, ())

    assert result.output == -3.0


def test_non_finite_operator_result_is_normalized_to_zero() -> None:
    registry = OperatorRegistry(
        (Operator(OperatorName("huge"), 1, lambda values: float("inf")),)
    )
    program = make_program(
        Instruction(
            OperatorName("huge"),
            RegisterIndex(0),
            (ConstantOperand(1.0),),
        )
    )

    result = ProgramExecutor(RuntimeConfig(0, 1), registry).execute(program, ())

    assert result.output == 0.0


def test_arithmetic_exception_is_normalized_to_zero() -> None:
    def fail(_values: tuple[float, ...]) -> float:
        raise OverflowError

    registry = OperatorRegistry((Operator(OperatorName("fail"), 1, fail),))
    program = make_program(
        Instruction(
            OperatorName("fail"),
            RegisterIndex(0),
            (ConstantOperand(1.0),),
        )
    )

    result = ProgramExecutor(RuntimeConfig(0, 1), registry).execute(program, ())

    assert result.output == 0.0


def test_non_real_custom_operator_result_is_an_error() -> None:
    def invalid(_values: tuple[float, ...]) -> float:
        return "not a number"  # type: ignore[return-value]

    registry = OperatorRegistry((Operator(OperatorName("invalid"), 1, invalid),))
    program = make_program(
        Instruction(
            OperatorName("invalid"),
            RegisterIndex(0),
            (ConstantOperand(1.0),),
        )
    )

    with pytest.raises(OperatorExecutionError, match="non-real"):
        ProgramExecutor(RuntimeConfig(0, 1), registry).execute(program, ())


@pytest.mark.parametrize("observation", [(1.0,), (1.0, 2.0, 3.0)])
def test_observation_length_must_match_configuration(
    observation: tuple[float, ...],
) -> None:
    program = make_program(
        Instruction(
            OperatorName("identity"),
            RegisterIndex(0),
            (ConstantOperand(0.0),),
        )
    )

    with pytest.raises(InvalidObservationError, match="length"):
        ProgramExecutor(RuntimeConfig(2, 1)).execute(program, observation)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_observation_values_must_be_finite_reals(value: float) -> None:
    program = make_program(
        Instruction(
            OperatorName("identity"),
            RegisterIndex(0),
            (InputOperand(InputIndex(0)),),
        )
    )

    with pytest.raises(InvalidObservationError, match="real number|finite"):
        ProgramExecutor(RuntimeConfig(1, 1)).execute(program, (value,))


def test_unknown_operator_is_rejected() -> None:
    program = make_program(
        Instruction(
            OperatorName("missing"),
            RegisterIndex(0),
            (ConstantOperand(0.0),),
        )
    )

    with pytest.raises(UnknownOperatorError, match="missing"):
        ProgramExecutor(RuntimeConfig(0, 1)).validate(program)


def test_wrong_operator_arity_is_rejected() -> None:
    program = make_program(
        Instruction(
            OperatorName("add"),
            RegisterIndex(0),
            (ConstantOperand(1.0),),
        )
    )

    with pytest.raises(OperatorArityError, match="expects 2 operands"):
        ProgramExecutor(RuntimeConfig(0, 1)).validate(program)


@pytest.mark.parametrize(
    "instruction",
    [
        Instruction(
            OperatorName("identity"),
            RegisterIndex(1),
            (ConstantOperand(0.0),),
        ),
        Instruction(
            OperatorName("identity"),
            RegisterIndex(0),
            (RegisterOperand(RegisterIndex(1)),),
        ),
        Instruction(
            OperatorName("identity"),
            RegisterIndex(0),
            (InputOperand(InputIndex(1)),),
        ),
    ],
)
def test_configured_bounds_are_strict(instruction: Instruction) -> None:
    with pytest.raises(InvalidInstructionError, match="outside"):
        ProgramExecutor(RuntimeConfig(1, 1)).validate(make_program(instruction))


@pytest.mark.parametrize(
    ("input_size", "register_count"),
    [(-1, 1), (True, 1), (0, 0), (0, -1), (0, True)],
)
def test_runtime_configuration_rejects_invalid_sizes(
    input_size: int,
    register_count: int,
) -> None:
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig(input_size, register_count)
