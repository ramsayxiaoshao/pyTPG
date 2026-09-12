"""Call-order-independent named random streams for recorded experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pytpg.evolution.errors import EvolutionConfigurationError
from pytpg.evolution.rng import RandomGenerator, create_rng


@dataclass(frozen=True, slots=True)
class SeedManager:
    """Derive deterministic named RNG streams from one recorded master seed."""

    master_seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.master_seed, bool)
            or not isinstance(self.master_seed, int)
            or self.master_seed < 0
        ):
            raise EvolutionConfigurationError(
                "master_seed must be a non-negative integer"
            )

    def derive_seed(self, namespace: str, index: int = 0) -> int:
        """Derive a stable 128-bit seed independently of request order."""

        if not isinstance(namespace, str) or not namespace.strip():
            raise EvolutionConfigurationError(
                "seed namespace must be a non-empty string"
            )
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise EvolutionConfigurationError("seed index must be non-negative")
        payload = (
            f"pytpg-seed-v1\0{self.master_seed}\0{namespace}\0{index}"
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")

    def rng(self, namespace: str, index: int = 0) -> RandomGenerator:
        """Create a fresh NumPy generator for one named deterministic stream."""

        return create_rng(self.derive_seed(namespace, index))


__all__ = ["SeedManager"]
