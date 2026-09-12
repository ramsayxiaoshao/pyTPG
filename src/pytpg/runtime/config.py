"""Configuration for deterministic program execution."""

from dataclasses import dataclass

from pytpg.runtime.errors import RuntimeConfigurationError


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Context required to validate and execute a program.

    An input size of zero is valid for programs that use only constants and
    registers. At least one register is required because a program always emits
    its bid through an output register.
    """

    input_size: int
    register_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_size, bool)
            or not isinstance(self.input_size, int)
            or self.input_size < 0
        ):
            msg = f"input_size must be a non-negative integer, got {self.input_size!r}"
            raise RuntimeConfigurationError(msg)
        if (
            isinstance(self.register_count, bool)
            or not isinstance(self.register_count, int)
            or self.register_count < 1
        ):
            msg = (
                "register_count must be a positive integer, "
                f"got {self.register_count!r}"
            )
            raise RuntimeConfigurationError(msg)


__all__ = ["RuntimeConfig"]
