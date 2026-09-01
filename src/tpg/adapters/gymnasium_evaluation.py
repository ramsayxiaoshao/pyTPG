"""Reproducible episodic Gymnasium fitness evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tpg.adapters.errors import GymnasiumEvaluationError
from tpg.adapters.gymnasium import (
    GymnasiumAdapter,
    GymnasiumEnvironmentFactory,
)
from tpg.core import TPGGraph
from tpg.runtime import GraphRuntime
from tpg.seed import SeedManager


@dataclass(frozen=True, slots=True)
class GymnasiumEvaluationConfig:
    """Episode seeds and hard rollout bound shared by every graph."""

    episode_seeds: tuple[int, ...]
    max_episode_steps: int = 1_000

    def __post_init__(self) -> None:
        if not isinstance(self.episode_seeds, tuple) or not self.episode_seeds:
            raise GymnasiumEvaluationError("episode_seeds must be a non-empty tuple")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.episode_seeds
        ):
            raise GymnasiumEvaluationError(
                "episode seeds must be non-negative integers"
            )
        if (
            isinstance(self.max_episode_steps, bool)
            or not isinstance(self.max_episode_steps, int)
            or self.max_episode_steps < 1
        ):
            raise GymnasiumEvaluationError(
                "max_episode_steps must be a positive integer"
            )

    @classmethod
    def from_seed(
        cls,
        master_seed: int,
        episodes: int,
        *,
        max_episode_steps: int = 1_000,
    ) -> GymnasiumEvaluationConfig:
        """Derive call-order-independent common episode seeds."""

        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise GymnasiumEvaluationError("episodes must be a positive integer")
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or master_seed < 0
        ):
            raise GymnasiumEvaluationError("master_seed must be a non-negative integer")
        seed_manager = SeedManager(master_seed)
        return cls(
            tuple(
                seed_manager.derive_seed("gymnasium_episode", index)
                for index in range(episodes)
            ),
            max_episode_steps,
        )


@dataclass(frozen=True, slots=True)
class GymnasiumEpisodeResult:
    """Inspectable return and termination reason for one rollout."""

    seed: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    step_limit_reached: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise GymnasiumEvaluationError("episode seed must be non-negative")
        if (
            isinstance(self.total_reward, bool)
            or not isinstance(self.total_reward, (int, float))
            or not math.isfinite(float(self.total_reward))
        ):
            raise GymnasiumEvaluationError("episode total_reward must be finite")
        object.__setattr__(self, "total_reward", float(self.total_reward))
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps < 1
        ):
            raise GymnasiumEvaluationError("episode steps must be positive")
        if not all(
            isinstance(value, bool)
            for value in (self.terminated, self.truncated, self.step_limit_reached)
        ):
            raise GymnasiumEvaluationError("episode termination flags must be bools")
        if self.step_limit_reached == (self.terminated or self.truncated):
            raise GymnasiumEvaluationError(
                "step_limit_reached must be the complement of environment completion"
            )


class GymnasiumFitness:
    """Mean episodic return with common random seeds across all graphs."""

    __slots__ = ("config", "environment_factory")

    def __init__(
        self,
        environment_factory: GymnasiumEnvironmentFactory,
        config: GymnasiumEvaluationConfig,
    ) -> None:
        if not callable(environment_factory):
            raise GymnasiumEvaluationError("environment_factory must be callable")
        if not isinstance(config, GymnasiumEvaluationConfig):
            raise GymnasiumEvaluationError("config must be a GymnasiumEvaluationConfig")
        self.environment_factory = environment_factory
        self.config = config

    def __call__(self, graph: TPGGraph) -> float:
        """Return mean total reward, suitable for `SequentialEvaluator`."""

        results = self.evaluate(graph)
        return math.fsum(result.total_reward for result in results) / len(results)

    def evaluate(self, graph: TPGGraph) -> tuple[GymnasiumEpisodeResult, ...]:
        """Evaluate all configured episodes in one fresh, always-closed env."""

        if not isinstance(graph, TPGGraph):
            raise GymnasiumEvaluationError("graph must be a TPGGraph")
        adapter = GymnasiumAdapter(self.environment_factory())
        with adapter:
            self._validate_graph_actions(graph, adapter.n_actions)
            runtime: GraphRuntime | None = None
            results: list[GymnasiumEpisodeResult] = []
            for seed in self.config.episode_seeds:
                reset = adapter.reset(seed=seed)
                if runtime is None:
                    runtime = GraphRuntime.infer_for_graph(
                        graph,
                        reset.observation,
                        validate_before_execution=False,
                    )
                    runtime.validate(graph).require_valid()
                total_reward = 0.0
                observation = reset.observation
                terminated = False
                truncated = False
                steps = 0
                for _ in range(self.config.max_episode_steps):
                    action_id = runtime.act(graph, observation)
                    transition = adapter.step(action_id)
                    steps += 1
                    total_reward += transition.reward
                    if not math.isfinite(total_reward):
                        raise GymnasiumEvaluationError(
                            "accumulated episode reward is not finite"
                        )
                    observation = transition.observation
                    terminated = transition.terminated
                    truncated = transition.truncated
                    if transition.done:
                        break
                step_limit_reached = not (terminated or truncated)
                results.append(
                    GymnasiumEpisodeResult(
                        seed,
                        total_reward,
                        steps,
                        terminated,
                        truncated,
                        step_limit_reached,
                    )
                )
        return tuple(results)

    @staticmethod
    def _validate_graph_actions(graph: TPGGraph, n_actions: int) -> None:
        for team in graph.teams:
            for learner in team.learners:
                if learner.action.kind != "atomic":
                    continue
                action_id = int(learner.action.action_id)
                if not 0 <= action_id < n_actions:
                    raise GymnasiumEvaluationError(
                        f"graph atomic action ID {action_id} is outside "
                        f"the environment range [0, {n_actions})"
                    )


__all__ = [
    "GymnasiumEpisodeResult",
    "GymnasiumEvaluationConfig",
    "GymnasiumFitness",
]
