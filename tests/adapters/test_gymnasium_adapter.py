"""Unit tests for the dependency-free Gymnasium adapter boundary."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from tpg.adapters import (
    GymnasiumAdapter,
    GymnasiumDependencyError,
    GymnasiumEnvironment,
    GymnasiumEpisodeStateError,
    InvalidGymnasiumActionError,
    InvalidGymnasiumTransitionError,
    UnsupportedGymnasiumSpaceError,
    make_gymnasium_environment,
)


@dataclass
class DiscreteSpace:
    n: int
    start: int = 0
    shape: tuple[()] = ()

    def contains(self, value: object) -> bool:
        return isinstance(value, int) and self.start <= value < self.start + self.n


@dataclass
class NumericSpace:
    shape: tuple[int, ...]
    dtype: object = np.dtype("float64")


class MockEnvironment:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (2, 2),
        action_start: int = 3,
    ) -> None:
        self.action_space = DiscreteSpace(2, action_start)
        self.observation_space = NumericSpace(shape)
        self.reset_calls: list[tuple[int | None, Mapping[str, object] | None]] = []
        self.actions: list[object] = []
        self.close_calls = 0
        self.step_result: object = (
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            1.5,
            False,
            True,
            {"cause": "limit"},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> object:
        self.reset_calls.append((seed, options))
        if self.observation_space.shape == ():
            observation: object = np.int64(4)
        else:
            observation = np.array([[1.0, 2.0], [3.0, 4.0]])
        return observation, {"seed": seed}

    def step(self, action: object) -> object:
        self.actions.append(action)
        return self.step_result

    def close(self) -> None:
        self.close_calls += 1


def adapter_for(environment: MockEnvironment) -> GymnasiumAdapter:
    return GymnasiumAdapter(cast(GymnasiumEnvironment, environment))


def test_reset_flattens_numeric_observation_and_preserves_info() -> None:
    environment = MockEnvironment()
    adapter = adapter_for(environment)

    result = adapter.reset(seed=17, options={"difficulty": "test"})

    assert adapter.input_size == 4
    assert adapter.n_actions == 2
    assert result.observation == (1.0, 2.0, 3.0, 4.0)
    assert result.info == {"seed": 17}
    assert environment.reset_calls == [(17, {"difficulty": "test"})]


def test_action_id_maps_to_discrete_start_and_step_uses_modern_flags() -> None:
    environment = MockEnvironment(action_start=3)
    adapter = adapter_for(environment)
    adapter.reset(seed=0)

    result = adapter.step(1)

    assert environment.actions == [4]
    assert result.environment_action == 4
    assert result.observation == (5.0, 6.0, 7.0, 8.0)
    assert result.reward == 1.5
    assert result.terminated is False
    assert result.truncated is True
    assert result.done is True
    assert result.info == {"cause": "limit"}


def test_scalar_numeric_observation_is_a_one_value_vector() -> None:
    environment = MockEnvironment(shape=(), action_start=0)
    adapter = adapter_for(environment)

    result = adapter.reset(seed=2)

    assert adapter.input_size == 1
    assert result.observation == (4.0,)


def test_episode_lifecycle_requires_reset_and_close_is_idempotent() -> None:
    environment = MockEnvironment()
    adapter = adapter_for(environment)

    with pytest.raises(GymnasiumEpisodeStateError, match="reset"):
        adapter.step(0)
    adapter.reset()
    adapter.step(0)
    with pytest.raises(GymnasiumEpisodeStateError, match="reset"):
        adapter.step(0)
    adapter.close()
    adapter.close()
    with pytest.raises(GymnasiumEpisodeStateError, match="closed"):
        adapter.reset()

    assert environment.close_calls == 1


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_reset_seed_is_rejected(seed: object) -> None:
    adapter = adapter_for(MockEnvironment())

    with pytest.raises(InvalidGymnasiumTransitionError, match="seed"):
        adapter.reset(seed=seed)  # type: ignore[arg-type]


def test_invalid_reset_options_are_rejected() -> None:
    adapter = adapter_for(MockEnvironment())

    with pytest.raises(InvalidGymnasiumTransitionError, match="options"):
        adapter.reset(options=[])  # type: ignore[arg-type]
    with pytest.raises(InvalidGymnasiumTransitionError, match="keys"):
        adapter.reset(options=cast(Mapping[str, object], {1: "invalid"}))


@pytest.mark.parametrize("action_id", [-1, 2, True, 1.5])
def test_invalid_tpg_actions_are_rejected(action_id: object) -> None:
    adapter = adapter_for(MockEnvironment(action_start=0))

    with pytest.raises(InvalidGymnasiumActionError):
        adapter.environment_action(action_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("action_space", "observation_space"),
    [
        (object(), NumericSpace((1,))),
        (DiscreteSpace(0), NumericSpace((1,))),
        (DiscreteSpace(2, shape=(1,)), NumericSpace((1,))),
        (DiscreteSpace(2), object()),
        (DiscreteSpace(2), NumericSpace((0,))),
        (DiscreteSpace(2), NumericSpace((1,), np.dtype("bool"))),
    ],
)
def test_unsupported_spaces_are_rejected(
    action_space: object,
    observation_space: object,
) -> None:
    environment = MockEnvironment()
    environment.action_space = cast(DiscreteSpace, action_space)
    environment.observation_space = cast(NumericSpace, observation_space)

    with pytest.raises(UnsupportedGymnasiumSpaceError):
        adapter_for(environment)


@pytest.mark.parametrize(
    ("operation", "result", "message"),
    [
        ("reset", [np.zeros((2, 2)), {}], "2-item tuple"),
        ("reset", (np.zeros(4), {}), "shape"),
        ("reset", (np.full((2, 2), np.nan), {}), "finite"),
        ("reset", (np.zeros((2, 2)), []), "mapping"),
        ("step", (np.zeros((2, 2)), 1.0, False, {}), "5-item tuple"),
        ("step", (np.zeros((2, 2)), float("inf"), False, False, {}), "finite"),
        ("step", (np.zeros((2, 2)), 1.0, 0, False, {}), "terminated"),
        ("step", (np.zeros((2, 2)), 1.0, False, False, []), "mapping"),
    ],
)
def test_malformed_environment_results_are_rejected(
    operation: str,
    result: object,
    message: str,
) -> None:
    environment = MockEnvironment()
    adapter = adapter_for(environment)
    if operation == "reset":
        environment.reset = lambda **_kwargs: result  # type: ignore[method-assign]
        with pytest.raises(InvalidGymnasiumTransitionError, match=message):
            adapter.reset()
    else:
        adapter.reset()
        environment.step_result = result
        with pytest.raises(InvalidGymnasiumTransitionError, match=message):
            adapter.step(0)

        with pytest.raises(GymnasiumEpisodeStateError, match="reset"):
            adapter.step(0)


def test_factory_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.import_module

    def missing(name: str, package: str | None = None) -> object:
        if name == "gymnasium":
            raise ModuleNotFoundError("missing", name="gymnasium")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(GymnasiumDependencyError, match="extra"):
        make_gymnasium_environment("CartPole-v1")
