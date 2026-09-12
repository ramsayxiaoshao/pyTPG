"""Exact JSON-compatible snapshots for the reference NumPy RNG."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any, cast

from pytpg.evolution.rng import RandomGenerator
from pytpg.serialization._validation import exact_keys, mapping, string
from pytpg.serialization.errors import (
    CheckpointCompatibilityError,
    InvalidSerializedDataError,
)

REFERENCE_BIT_GENERATOR = "PCG64"


@dataclass(frozen=True, slots=True)
class RNGSnapshot:
    """Canonical JSON state for the reference PCG64 generator."""

    bit_generator: str
    state_json: str

    def __post_init__(self) -> None:
        if self.bit_generator != REFERENCE_BIT_GENERATOR:
            raise CheckpointCompatibilityError(
                f"unsupported bit generator {self.bit_generator!r}"
            )
        try:
            state = json.loads(self.state_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise InvalidSerializedDataError("RNG state is not valid JSON") from error
        if not isinstance(state, dict):
            raise InvalidSerializedDataError("RNG state must be a JSON object")

    def to_dict(self) -> dict[str, object]:
        return {
            "bit_generator": self.bit_generator,
            "state": json.loads(self.state_json),
        }

    @classmethod
    def from_dict(cls, value: object) -> RNGSnapshot:
        item = mapping(value, "rng")
        exact_keys(item, {"bit_generator", "state"}, "rng")
        try:
            state_json = json.dumps(
                item["state"], sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise InvalidSerializedDataError(
                "RNG state is not JSON-compatible"
            ) from error
        return cls(string(item["bit_generator"], "rng.bit_generator"), state_json)


def capture_rng(rng: RandomGenerator) -> RNGSnapshot:
    """Capture the exact state of a generator created by `create_rng`."""

    bit_generator: Any = getattr(rng, "bit_generator", None)
    if bit_generator is None:
        raise CheckpointCompatibilityError(
            "RNG does not expose a NumPy bit_generator state"
        )
    name = type(bit_generator).__name__
    if name != REFERENCE_BIT_GENERATOR:
        raise CheckpointCompatibilityError(f"unsupported bit generator {name!r}")
    try:
        state_json = json.dumps(
            bit_generator.state,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            "bit generator state is not JSON-compatible"
        ) from error
    return RNGSnapshot(name, state_json)


def restore_rng(snapshot: RNGSnapshot) -> RandomGenerator:
    """Restore a fresh generator whose next draw exactly matches the snapshot."""

    if not isinstance(snapshot, RNGSnapshot):
        raise CheckpointCompatibilityError("snapshot must be an RNGSnapshot")
    numpy: Any = importlib.import_module("numpy")
    bit_generator: Any = numpy.random.PCG64()
    try:
        bit_generator.state = json.loads(snapshot.state_json)
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidSerializedDataError("RNG state is invalid for PCG64") from error
    return cast(RandomGenerator, numpy.random.Generator(bit_generator))


__all__ = [
    "REFERENCE_BIT_GENERATOR",
    "RNGSnapshot",
    "capture_rng",
    "restore_rng",
]
