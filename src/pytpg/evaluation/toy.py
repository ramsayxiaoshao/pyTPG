"""Small deterministic tasks for runtime and evolutionary integration tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pytpg.core import ActionID, TPGGraph
from pytpg.evolution.errors import EvolutionConfigurationError


@dataclass(frozen=True, slots=True)
class ContextualBanditCase:
    """One normalized observation and its desired atomic action."""

    observation: tuple[float, ...]
    expected_action: ActionID

    def __post_init__(self) -> None:
        if not isinstance(self.observation, tuple):
            raise EvolutionConfigurationError("case observation must be a tuple")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.observation
        ):
            raise EvolutionConfigurationError(
                "case observations must contain only finite real values"
            )
        if (
            isinstance(self.expected_action, bool)
            or not isinstance(self.expected_action, int)
            or self.expected_action < 0
        ):
            raise EvolutionConfigurationError(
                "case expected action must be a non-negative integer ID"
            )


@dataclass(frozen=True, slots=True)
class ContextualBandit:
    """Accuracy objective over a finite deterministic classification table."""

    input_size: int
    n_actions: int
    cases: tuple[ContextualBanditCase, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_size, bool)
            or not isinstance(self.input_size, int)
            or self.input_size < 0
        ):
            raise EvolutionConfigurationError(
                "input_size must be a non-negative integer"
            )
        if (
            isinstance(self.n_actions, bool)
            or not isinstance(self.n_actions, int)
            or self.n_actions < 1
        ):
            raise EvolutionConfigurationError("n_actions must be a positive integer")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise EvolutionConfigurationError("contextual bandit requires cases")
        for case in self.cases:
            if not isinstance(case, ContextualBanditCase):
                raise EvolutionConfigurationError(
                    "contextual bandit cases must be ContextualBanditCase values"
                )
            if len(case.observation) != self.input_size:
                raise EvolutionConfigurationError(
                    "case observation length does not match input_size"
                )
            if not 0 <= case.expected_action < self.n_actions:
                raise EvolutionConfigurationError(
                    "expected action is outside the task action space"
                )

    def __call__(self, graph: TPGGraph) -> float:
        """Return exact classification accuracy in the closed interval [0, 1]."""

        correct = sum(
            graph.act(case.observation) == case.expected_action for case in self.cases
        )
        return correct / len(self.cases)

    @classmethod
    def signed_binary(cls) -> ContextualBandit:
        """Return `negative -> 0`, `positive -> 1` classification."""

        return cls(
            input_size=1,
            n_actions=2,
            cases=(
                ContextualBanditCase((-1.0,), ActionID(0)),
                ContextualBanditCase((1.0,), ActionID(1)),
            ),
        )


__all__ = ["ContextualBandit", "ContextualBanditCase"]
