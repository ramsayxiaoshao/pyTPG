"""Context-free validation helpers for core value objects."""


def require_non_negative_integer(value: int, name: str) -> None:
    """Raise when an ID or index is not a non-negative plain integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{name} must be a non-negative integer, got {value!r}"
        raise ValueError(msg)
