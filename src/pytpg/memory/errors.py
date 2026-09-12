"""Stateful-memory configuration and lifecycle failures."""


class TPGMemoryError(Exception):
    """Base class for memory composition failures."""


class MemoryConfigurationError(TPGMemoryError):
    """A memory component or stateful runtime is configured inconsistently."""


class MemoryStateError(TPGMemoryError):
    """A memory value, snapshot, update, or episode transition is invalid."""


__all__ = ["MemoryConfigurationError", "MemoryStateError", "TPGMemoryError"]
