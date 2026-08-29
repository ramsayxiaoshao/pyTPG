"""Explicit NumPy random-generator construction and narrow RNG protocol."""

import importlib
from collections.abc import Iterable
from typing import Any, Protocol, SupportsFloat, SupportsInt, cast

from tpg.evolution.errors import EvolutionConfigurationError


class RandomGenerator(Protocol):
    """The narrow `numpy.random.Generator` surface used by evolution."""

    def integers(self, high: int) -> SupportsInt: ...

    def uniform(self, low: float, high: float) -> SupportsFloat: ...

    def choice(
        self,
        values: int,
        *,
        size: int,
        replace: bool,
    ) -> Iterable[SupportsInt]: ...

    def random(self) -> SupportsFloat: ...


def create_rng(seed: int) -> RandomGenerator:
    """Create the sole RNG stream for a reproducible evolution run."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        msg = f"seed must be a non-negative integer, got {seed!r}"
        raise EvolutionConfigurationError(msg)
    numpy: Any = importlib.import_module("numpy")
    return cast(RandomGenerator, numpy.random.default_rng(seed))


__all__ = ["RandomGenerator", "create_rng"]
