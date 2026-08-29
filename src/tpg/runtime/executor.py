"""Deterministic reference executor for one TPG program."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tpg.runtime._model import (
    InstructionLike,
    OperandLike,
    ProgramLike,
)
from tpg.runtime.config import RuntimeConfig
from tpg.runtime.errors import (
    InvalidInstructionError,
    InvalidObservationError,
    OperatorArityError,
    OperatorExecutionError,
)
from tpg.runtime.operators import Operator, OperatorRegistry, default_operator_registry
from tpg.runtime.registers import RegisterFile


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """A program bid together with its final register state."""

    output: float
    registers: tuple[float, ...]


class ProgramExecutor:
    """Validate and execute programs under one explicit runtime configuration."""

    __slots__ = ("config", "operators")

    def __init__(
        self,
        config: RuntimeConfig,
        operators: OperatorRegistry | None = None,
    ) -> None:
        self.config = config
        self.operators = operators or default_operator_registry()

    def validate(self, program: ProgramLike) -> None:
        """Validate all configuration-dependent instruction constraints."""

        for position, instruction in enumerate(program.instructions):
            self.validate_instruction(instruction, position)

    def validate_instruction(
        self,
        instruction: InstructionLike,
        position: int = 0,
    ) -> None:
        """Validate one instruction and retain its program position in errors."""

        operator = self.operators.resolve(instruction.operator)
        if len(instruction.operands) != operator.arity:
            msg = (
                f"instruction {position} operator {operator.name!r} expects "
                f"{operator.arity} operands, got {len(instruction.operands)}"
            )
            raise OperatorArityError(msg)
        if instruction.destination >= self.config.register_count:
            msg = (
                f"instruction {position} destination register "
                f"{instruction.destination} is outside "
                f"[0, {self.config.register_count})"
            )
            raise InvalidInstructionError(msg)

        for operand in instruction.operands:
            if operand.kind == "input" and operand.index >= self.config.input_size:
                msg = (
                    f"instruction {position} input index {operand.index} is outside "
                    f"[0, {self.config.input_size})"
                )
                raise InvalidInstructionError(msg)
            if (
                operand.kind == "register"
                and operand.index >= self.config.register_count
            ):
                msg = (
                    f"instruction {position} register operand {operand.index} is "
                    f"outside [0, {self.config.register_count})"
                )
                raise InvalidInstructionError(msg)

    def execute(
        self,
        program: ProgramLike,
        observation: Sequence[float],
    ) -> ExecutionResult:
        """Execute with fresh zero registers and return a finite raw bid."""

        normalized_observation = self.normalize_observation(observation)
        self.validate(program)
        registers = RegisterFile(self.config.register_count)

        for instruction in program.instructions:
            operator = self.operators.resolve(instruction.operator)
            operands = tuple(
                self._resolve_operand(operand, normalized_observation, registers)
                for operand in instruction.operands
            )
            result = self._apply_operator(operator, operands)
            registers.write(instruction.destination, result)

        return ExecutionResult(
            output=registers.read(0),
            registers=registers.snapshot(),
        )

    def normalize_observation(
        self,
        observation: Sequence[float],
    ) -> tuple[float, ...]:
        """Validate observation shape and convert values to finite floats."""

        if isinstance(observation, (str, bytes)):
            raise InvalidObservationError("observation must be a numeric sequence")
        try:
            values = tuple(observation)
        except TypeError as error:
            raise InvalidObservationError("observation must be a sequence") from error
        if len(values) != self.config.input_size:
            msg = (
                f"observation length {len(values)} does not match configured "
                f"input_size {self.config.input_size}"
            )
            raise InvalidObservationError(msg)

        normalized: list[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"observation[{index}] must be a real number, got {value!r}"
                raise InvalidObservationError(msg)
            converted = float(value)
            if not math.isfinite(converted):
                msg = f"observation[{index}] must be finite, got {value!r}"
                raise InvalidObservationError(msg)
            normalized.append(converted)
        return tuple(normalized)

    @staticmethod
    def _resolve_operand(
        operand: OperandLike,
        observation: tuple[float, ...],
        registers: RegisterFile,
    ) -> float:
        if operand.kind == "input":
            return observation[operand.index]
        if operand.kind == "register":
            return registers.read(operand.index)
        return float(operand.value)

    @staticmethod
    def _apply_operator(operator: Operator, values: tuple[float, ...]) -> float:
        try:
            result = operator.function(values)
        except (ArithmeticError, ValueError):
            return 0.0
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            msg = f"operator {operator.name!r} returned a non-real value {result!r}"
            raise OperatorExecutionError(msg)
        normalized = float(result)
        return normalized if math.isfinite(normalized) else 0.0


__all__ = ["ExecutionResult", "ProgramExecutor"]
