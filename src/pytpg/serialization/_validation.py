"""Small strict helpers shared by JSON schema readers."""

from __future__ import annotations

import math
from typing import Any, cast

from pytpg.serialization.errors import InvalidSerializedDataError


def mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidSerializedDataError(f"{location} must be an object")
    untyped = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in untyped):
        raise InvalidSerializedDataError(f"{location} keys must be strings")
    return cast(dict[str, Any], untyped)


def sequence(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidSerializedDataError(f"{location} must be an array")
    return cast(list[Any], value)


def exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise InvalidSerializedDataError(
            f"{location} fields must be exactly {sorted(expected)}"
        )


def integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidSerializedDataError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def finite_number(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise InvalidSerializedDataError(f"{location} must be a finite number")
    return float(value)


def string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSerializedDataError(f"{location} must be a non-empty string")
    return value


def boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidSerializedDataError(f"{location} must be boolean")
    return value


__all__ = [
    "boolean",
    "exact_keys",
    "finite_number",
    "integer",
    "mapping",
    "sequence",
    "string",
]
