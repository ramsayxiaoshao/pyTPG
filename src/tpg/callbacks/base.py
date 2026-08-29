"""Structured evolution lifecycle events and callback protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

EvolutionEventKind: TypeAlias = Literal[
    "run_started",
    "generation_evaluated",
    "generation_reproduced",
    "run_finished",
]


@dataclass(frozen=True, slots=True)
class EvolutionEvent:
    """Environment-independent, JSON-compatible lifecycle event."""

    kind: EvolutionEventKind
    generation: int
    population_size: int
    resumed: bool = False
    best_fitness: float | None = None
    elite_count: int = 0
    mutation_attempts: int = 0
    applied_mutations: int = 0

    def __post_init__(self) -> None:
        if self.kind not in (
            "run_started",
            "generation_evaluated",
            "generation_reproduced",
            "run_finished",
        ):
            raise ValueError("unsupported evolution event kind")
        for name, value in (
            ("generation", self.generation),
            ("population_size", self.population_size),
            ("elite_count", self.elite_count),
            ("mutation_attempts", self.mutation_attempts),
            ("applied_mutations", self.applied_mutations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"event {name} must be a non-negative integer"
                )
        if self.population_size < 1:
            raise ValueError(
                "event population_size must be positive"
            )
        if not isinstance(self.resumed, bool):
            raise ValueError("event resumed must be boolean")
        if self.best_fitness is not None and (
            isinstance(self.best_fitness, bool)
            or not isinstance(self.best_fitness, (int, float))
            or not math.isfinite(float(self.best_fitness))
        ):
            raise ValueError(
                "event best_fitness must be finite when present"
            )
        if self.applied_mutations > self.mutation_attempts:
            raise ValueError(
                "applied mutations cannot exceed mutation attempts"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the stable field representation used by structured loggers."""

        return {
            "kind": self.kind,
            "generation": self.generation,
            "population_size": self.population_size,
            "resumed": self.resumed,
            "best_fitness": self.best_fitness,
            "elite_count": self.elite_count,
            "mutation_attempts": self.mutation_attempts,
            "applied_mutations": self.applied_mutations,
        }


class EvolutionCallback(Protocol):
    """Receive ordered lifecycle events; callback errors propagate."""

    def on_event(self, event: EvolutionEvent) -> None: ...


def dispatch_event(
    callbacks: tuple[EvolutionCallback, ...],
    event: EvolutionEvent,
) -> None:
    """Deliver one event in stable callback order."""

    for callback in callbacks:
        callback.on_event(event)


__all__ = [
    "EvolutionCallback",
    "EvolutionEvent",
    "EvolutionEventKind",
    "dispatch_event",
]
