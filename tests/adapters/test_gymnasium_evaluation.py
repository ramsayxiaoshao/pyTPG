"""Fitness and real-environment integration tests for Gymnasium."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from tpg.adapters import (
    GymnasiumAdapter,
    GymnasiumEnvironment,
    GymnasiumEpisodeStateError,
    GymnasiumEvaluationConfig,
    GymnasiumEvaluationError,
    GymnasiumFitness,
    make_gymnasium_environment,
)
from tpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
    InputIndex,
    InputOperand,
    Instruction,
    Learner,
    LearnerID,
    OperatorName,
    Program,
    ProgramID,
    RegisterIndex,
    Team,
    TeamID,
    TPGGraph,
)
from tpg.memory import ObservationHistoryMemory
from tpg.runtime import GraphValidationError


@dataclass
class DiscreteSpace:
    n: int = 2
    start: int = 0
    shape: tuple[()] = ()

    def contains(self, value: object) -> bool:
        return isinstance(value, int) and self.start <= value < self.start + self.n


@dataclass
class NumericSpace:
    shape: tuple[int, ...] = (1,)
    dtype: object = np.dtype("float64")


class EpisodeEnvironment:
    action_space = DiscreteSpace()
    observation_space = NumericSpace()

    def __init__(self, seed_log: list[int | None], *, terminate_after: int = 2):
        self.seed_log = seed_log
        self.terminate_after = terminate_after
        self.steps = 0
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> object:
        del options
        self.seed_log.append(seed)
        self.steps = 0
        return np.array([float(seed or 0) % 7.0]), {}

    def step(self, action: object) -> object:
        self.steps += 1
        return (
            np.array([float(self.steps)]),
            float(cast(int, action) + 1),
            self.steps >= self.terminate_after,
            False,
            {},
        )

    def close(self) -> None:
        self.closed = True


class RecallEnvironment:
    action_space = DiscreteSpace()
    observation_space = NumericSpace()

    def __init__(self) -> None:
        self.signal = 0.0
        self.steps = 0
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> object:
        del options
        self.signal = 1.0 if cast(int, seed) % 2 == 0 else -1.0
        self.steps = 0
        return np.array([self.signal]), {}

    def step(self, action: object) -> object:
        self.steps += 1
        if self.steps == 1:
            return np.array([0.0]), 0.0, False, False, {}
        expected_action = 1 if self.signal > 0.0 else 0
        reward = 1.0 if action == expected_action else 0.0
        return np.array([0.0]), reward, True, False, {}

    def close(self) -> None:
        self.closed = True


def constant_graph(action_id: int) -> TPGGraph:
    instruction = Instruction(
        OperatorName("identity"),
        RegisterIndex(0),
        (ConstantOperand(1.0),),
    )
    learner = Learner(
        LearnerID(0),
        Program(ProgramID(0), (instruction,)),
        AtomicAction(ActionID(action_id)),
    )
    team = Team(TeamID(0), (learner,))
    return TPGGraph((team,), (team.id,))


def recall_graph() -> TPGGraph:
    negative = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("subtract"),
                RegisterIndex(0),
                (ConstantOperand(0.0), InputOperand(InputIndex(1))),
            ),
        ),
    )
    positive = Program(
        ProgramID(1),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (InputOperand(InputIndex(1)),),
            ),
        ),
    )
    team = Team(
        TeamID(0),
        (
            Learner(LearnerID(0), negative, AtomicAction(ActionID(0))),
            Learner(LearnerID(1), positive, AtomicAction(ActionID(1))),
        ),
    )
    return TPGGraph((team,), (team.id,))


def test_seed_derived_config_is_reproducible_and_call_order_independent() -> None:
    first = GymnasiumEvaluationConfig.from_seed(42, 3, max_episode_steps=9)
    second = GymnasiumEvaluationConfig.from_seed(42, 3, max_episode_steps=9)

    assert first == second
    assert len(first.episode_seeds) == 3
    assert len(set(first.episode_seeds)) == 3
    assert first.max_episode_steps == 9


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GymnasiumEvaluationConfig((), 1),
        lambda: GymnasiumEvaluationConfig((True,), 1),
        lambda: GymnasiumEvaluationConfig((0,), 0),
        lambda: GymnasiumEvaluationConfig.from_seed(-1, 1),
        lambda: GymnasiumEvaluationConfig.from_seed(1, 0),
    ],
)
def test_invalid_evaluation_configs_are_rejected(factory: object) -> None:
    with pytest.raises(GymnasiumEvaluationError):
        cast(object, factory)()  # type: ignore[operator]


def test_fitness_uses_common_seeds_reports_episodes_and_closes_envs() -> None:
    seed_log: list[int | None] = []
    environments: list[EpisodeEnvironment] = []

    def factory() -> GymnasiumEnvironment:
        environment = EpisodeEnvironment(seed_log)
        environments.append(environment)
        return cast(GymnasiumEnvironment, environment)

    config = GymnasiumEvaluationConfig((10, 20), max_episode_steps=5)
    fitness = GymnasiumFitness(factory, config)

    results = fitness.evaluate(constant_graph(1))
    score = fitness(constant_graph(1))

    assert tuple(result.seed for result in results) == (10, 20)
    assert tuple(result.total_reward for result in results) == (4.0, 4.0)
    assert tuple(result.steps for result in results) == (2, 2)
    assert all(result.terminated for result in results)
    assert not any(result.truncated for result in results)
    assert not any(result.step_limit_reached for result in results)
    assert score == 4.0
    assert seed_log == [10, 20, 10, 20]
    assert len(environments) == 2
    assert all(environment.closed for environment in environments)


def test_fitness_hard_step_limit_is_inspectable() -> None:
    environment = EpisodeEnvironment([], terminate_after=100)
    fitness = GymnasiumFitness(
        lambda: cast(GymnasiumEnvironment, environment),
        GymnasiumEvaluationConfig((1,), max_episode_steps=3),
    )

    result = fitness.evaluate(constant_graph(0))[0]

    assert result.steps == 3
    assert result.total_reward == 3.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.step_limit_reached is True
    assert environment.closed is True


def test_fitness_composes_fresh_episode_memory_without_rollout_duplication() -> None:
    environments: list[RecallEnvironment] = []
    memory_creations = 0

    def environment_factory() -> GymnasiumEnvironment:
        environment = RecallEnvironment()
        environments.append(environment)
        return cast(GymnasiumEnvironment, environment)

    def memory_factory() -> ObservationHistoryMemory:
        nonlocal memory_creations
        memory_creations += 1
        return ObservationHistoryMemory(1)

    fitness = GymnasiumFitness(
        environment_factory,
        GymnasiumEvaluationConfig((2, 3), max_episode_steps=2),
        memory_factory=memory_factory,
    )

    assert fitness(recall_graph()) == 1.0
    assert fitness(recall_graph()) == 1.0
    assert memory_creations == 2
    assert len(environments) == 2
    assert all(environment.closed for environment in environments)


def test_stateful_graph_requires_memory_channels_in_fitness() -> None:
    fitness = GymnasiumFitness(
        lambda: cast(GymnasiumEnvironment, RecallEnvironment()),
        GymnasiumEvaluationConfig((2,), max_episode_steps=2),
    )

    with pytest.raises(GraphValidationError, match="invalid_instruction"):
        fitness(recall_graph())


def test_fitness_rejects_non_callable_memory_factory() -> None:
    with pytest.raises(GymnasiumEvaluationError, match="memory_factory"):
        GymnasiumFitness(
            lambda: cast(GymnasiumEnvironment, RecallEnvironment()),
            GymnasiumEvaluationConfig((2,)),
            memory_factory=cast(object, 1),  # type: ignore[arg-type]
        )


def test_invalid_graph_action_closes_environment_before_failure() -> None:
    environment = EpisodeEnvironment([])
    fitness = GymnasiumFitness(
        lambda: cast(GymnasiumEnvironment, environment),
        GymnasiumEvaluationConfig((1,)),
    )

    with pytest.raises(GymnasiumEvaluationError, match="action ID"):
        fitness(constant_graph(2))

    assert environment.closed is True


def test_cartpole_adapter_and_fitness_are_deterministic_when_installed() -> None:
    pytest.importorskip("gymnasium")
    first = GymnasiumAdapter(make_gymnasium_environment("CartPole-v1"))
    second = GymnasiumAdapter(make_gymnasium_environment("CartPole-v1"))
    with first, second:
        first_reset = first.reset(seed=7)
        second_reset = second.reset(seed=7)
        assert first.input_size == 4
        assert first.n_actions == 2
        assert first_reset.observation == second_reset.observation

    config = GymnasiumEvaluationConfig((7, 11), max_episode_steps=10)
    fitness = GymnasiumFitness(
        lambda: make_gymnasium_environment("CartPole-v1"),
        config,
    )

    assert fitness(constant_graph(0)) == fitness(constant_graph(0))


def test_context_manager_cannot_reopen_closed_adapter() -> None:
    environment = EpisodeEnvironment([])
    adapter = GymnasiumAdapter(cast(GymnasiumEnvironment, environment))
    adapter.close()

    with pytest.raises(GymnasiumEpisodeStateError, match="closed"):
        with adapter:
            pass
