"""Random construction of valid instructions, programs, and atomic learners."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
    InputIndex,
    InputOperand,
    Instruction,
    Learner,
    LearnerID,
    OperatorName,
    Program,
    ProgramID,
    RegisterIndex,
    RegisterOperand,
)
from tpg.evolution.errors import EvolutionConfigurationError
from tpg.evolution.rng import RandomGenerator
from tpg.runtime import OperatorRegistry, RuntimeConfig, default_operator_registry


@dataclass(frozen=True, slots=True)
class GenomeConfig:
    """Shape and bounded random-construction policy for graph genomes."""

    input_size: int
    register_count: int
    n_actions: int
    min_program_length: int = 1
    max_program_length: int = 16
    constant_min: float = -1.0
    constant_max: float = 1.0

    def __post_init__(self) -> None:
        RuntimeConfig(self.input_size, self.register_count)
        if (
            isinstance(self.n_actions, bool)
            or not isinstance(self.n_actions, int)
            or self.n_actions < 1
        ):
            msg = f"n_actions must be a positive integer, got {self.n_actions!r}"
            raise EvolutionConfigurationError(msg)
        if (
            isinstance(self.min_program_length, bool)
            or not isinstance(self.min_program_length, int)
            or self.min_program_length < 1
        ):
            raise EvolutionConfigurationError(
                "min_program_length must be a positive integer"
            )
        if (
            isinstance(self.max_program_length, bool)
            or not isinstance(self.max_program_length, int)
            or self.max_program_length < self.min_program_length
        ):
            raise EvolutionConfigurationError(
                "max_program_length must be at least min_program_length"
            )
        constants = (self.constant_min, self.constant_max)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in constants
        ):
            raise EvolutionConfigurationError(
                "constant bounds must be finite real numbers"
            )
        if self.constant_min >= self.constant_max:
            raise EvolutionConfigurationError(
                "constant_min must be strictly less than constant_max"
            )


class GenomeFactory:
    """Construct random valid genome values using only a supplied RNG."""

    __slots__ = ("config", "operators")

    def __init__(
        self,
        config: GenomeConfig,
        operators: OperatorRegistry | None = None,
    ) -> None:
        self.config = config
        self.operators = operators or default_operator_registry()
        if not self.operators.names:
            raise EvolutionConfigurationError("operator registry cannot be empty")

    @property
    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(self.config.input_size, self.config.register_count)

    def random_instruction(
        self,
        rng: RandomGenerator,
        *,
        destination: RegisterIndex | None = None,
    ) -> Instruction:
        """Construct one registry-valid, in-bounds instruction."""

        name = self.operators.names[int(rng.integers(len(self.operators.names)))]
        operator = self.operators.resolve(name)
        destination_index = (
            RegisterIndex(int(rng.integers(self.config.register_count)))
            if destination is None
            else destination
        )
        return Instruction(
            OperatorName(name),
            destination_index,
            tuple(self.random_operand(rng) for _ in range(operator.arity)),
        )

    def random_operand(
        self,
        rng: RandomGenerator,
    ) -> InputOperand | RegisterOperand | ConstantOperand:
        """Choose uniformly from addressable operand variants."""

        kinds = (
            ("register", "constant")
            if self.config.input_size == 0
            else (
                "input",
                "register",
                "constant",
            )
        )
        kind = kinds[int(rng.integers(len(kinds)))]
        if kind == "input":
            return InputOperand(InputIndex(int(rng.integers(self.config.input_size))))
        if kind == "register":
            return RegisterOperand(
                RegisterIndex(int(rng.integers(self.config.register_count)))
            )
        return ConstantOperand(
            float(rng.uniform(self.config.constant_min, self.config.constant_max))
        )

    def random_program(
        self,
        program_id: ProgramID,
        length: int,
        rng: RandomGenerator,
    ) -> Program:
        """Construct a program whose final instruction writes bid register 0."""

        if (
            not self.config.min_program_length
            <= length
            <= self.config.max_program_length
        ):
            msg = (
                f"program length {length} is outside "
                f"[{self.config.min_program_length}, {self.config.max_program_length}]"
            )
            raise EvolutionConfigurationError(msg)
        instructions = [self.random_instruction(rng) for _ in range(length - 1)]
        instructions.append(self.random_instruction(rng, destination=RegisterIndex(0)))
        return Program(program_id, tuple(instructions))

    def random_atomic_learner(
        self,
        learner_id: LearnerID,
        program_id: ProgramID,
        program_length: int,
        rng: RandomGenerator,
    ) -> Learner:
        """Construct one learner with a valid random atomic action."""

        action_id = ActionID(int(rng.integers(self.config.n_actions)))
        return Learner(
            learner_id,
            self.random_program(program_id, program_length, rng),
            AtomicAction(action_id),
        )


__all__ = ["GenomeConfig", "GenomeFactory"]
