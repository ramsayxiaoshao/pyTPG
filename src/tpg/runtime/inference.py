"""Shape inference used only by low-level convenience APIs."""

from collections.abc import Iterable

from tpg.runtime._model import TeamLike
from tpg.runtime.config import RuntimeConfig


def infer_register_count(teams: Iterable[TeamLike]) -> int:
    """Return the smallest register file addressing every instruction."""

    register_count = 1
    for team in teams:
        for learner in team.learners:
            for instruction in learner.program.instructions:
                register_count = max(register_count, instruction.destination + 1)
                for operand in instruction.operands:
                    if operand.kind == "register":
                        register_count = max(register_count, operand.index + 1)
    return register_count


def infer_input_size(teams: Iterable[TeamLike]) -> int:
    """Return the smallest observation size addressing every input operand."""

    input_size = 0
    for team in teams:
        for learner in team.learners:
            for instruction in learner.program.instructions:
                for operand in instruction.operands:
                    if operand.kind == "input":
                        input_size = max(input_size, operand.index + 1)
    return input_size


def infer_validation_config(teams: Iterable[TeamLike]) -> RuntimeConfig:
    """Infer both dimensions when validating without experiment context."""

    materialized = tuple(teams)
    return RuntimeConfig(
        input_size=infer_input_size(materialized),
        register_count=infer_register_count(materialized),
    )


__all__ = ["infer_input_size", "infer_register_count", "infer_validation_config"]
