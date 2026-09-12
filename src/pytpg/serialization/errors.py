"""Specific diagnostics for versioned model and checkpoint data."""


class SerializationError(ValueError):
    """Base class for serialization and checkpoint failures."""


class InvalidSerializedDataError(SerializationError):
    """Raised when serialized input does not satisfy its declared schema."""


class UnsupportedFormatVersionError(SerializationError):
    """Raised when no reader exists for a declared format version."""


class CheckpointCompatibilityError(SerializationError):
    """Raised when checkpoint components cannot safely be resumed together."""


__all__ = [
    "CheckpointCompatibilityError",
    "InvalidSerializedDataError",
    "SerializationError",
    "UnsupportedFormatVersionError",
]
