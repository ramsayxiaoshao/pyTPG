"""Strict, optional boundary for modern Gymnasium environments."""

from __future__ import annotations

import importlib
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Protocol, cast

from tpg.adapters.errors import (
    GymnasiumDependencyError,
    GymnasiumEpisodeStateError,
    InvalidGymnasiumActionError,
    InvalidGymnasiumTransitionError,
    UnsupportedGymnasiumSpaceError,
)


class GymnasiumEnvironment(Protocol):
    """Structural surface used without importing the optional package."""

    action_space: object
    observation_space: object

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> object: ...

    def step(self, action: object) -> object: ...

    def close(self) -> None: ...


class GymnasiumEnvironmentFactory(Protocol):
    """Create a fresh environment for one graph fitness evaluation."""

    def __call__(self) -> GymnasiumEnvironment: ...


@dataclass(frozen=True, slots=True)
class GymnasiumReset:
    """Normalized result of one Gymnasium reset."""

    observation: tuple[float, ...]
    info: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class GymnasiumStep:
    """Normalized result of one Gymnasium step."""

    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, object]
    environment_action: int

    @property
    def done(self) -> bool:
        """Whether reset is required before another step."""

        return self.terminated or self.truncated


class GymnasiumAdapter:
    """Map flat numeric observations and zero-based TPG actions to Gymnasium."""

    __slots__ = (
        "_action_start",
        "_closed",
        "_needs_reset",
        "_observation_shape",
        "environment",
        "input_size",
        "n_actions",
    )

    def __init__(self, environment: GymnasiumEnvironment) -> None:
        self.environment = environment
        self.n_actions, self._action_start = self._validate_action_space(
            environment.action_space
        )
        self._observation_shape, self.input_size = self._validate_observation_space(
            environment.observation_space
        )
        self._closed = False
        self._needs_reset = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> GymnasiumReset:
        """Reset explicitly and normalize the `(observation, info)` result."""

        self._require_open()
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise InvalidGymnasiumTransitionError(
                "reset seed must be a non-negative integer or None"
            )
        if options is not None and not isinstance(options, Mapping):
            raise InvalidGymnasiumTransitionError(
                "reset options must be a mapping or None"
            )
        normalized_options: dict[str, object] | None = None
        if options is not None:
            untyped_options = cast(Mapping[object, object], options)
            if not all(isinstance(key, str) for key in untyped_options):
                raise InvalidGymnasiumTransitionError(
                    "reset option keys must all be strings"
                )
            normalized_options = cast(dict[str, object], dict(untyped_options))
        self._needs_reset = True
        result = self.environment.reset(seed=seed, options=normalized_options)
        values = self._require_result_tuple(result, 2, "reset")
        observation = self._normalize_observation(values[0])
        info = self._normalize_info(values[1], "reset info")
        self._needs_reset = False
        return GymnasiumReset(observation, info)

    def step(self, action_id: int) -> GymnasiumStep:
        """Map one action ID and normalize the five-value Gymnasium result."""

        self._require_open()
        if self._needs_reset:
            raise GymnasiumEpisodeStateError(
                "reset is required before stepping the environment"
            )
        environment_action = self.environment_action(action_id)
        self._needs_reset = True
        result = self.environment.step(environment_action)
        values = self._require_result_tuple(result, 5, "step")
        observation = self._normalize_observation(values[0])
        reward = self._normalize_reward(values[1])
        terminated = self._normalize_flag(values[2], "terminated")
        truncated = self._normalize_flag(values[3], "truncated")
        info = self._normalize_info(values[4], "step info")
        self._needs_reset = terminated or truncated
        return GymnasiumStep(
            observation,
            reward,
            terminated,
            truncated,
            info,
            environment_action,
        )

    def environment_action(self, action_id: int) -> int:
        """Translate a zero-based TPG action ID into `Discrete.start` space."""

        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise InvalidGymnasiumActionError(
                f"TPG action ID must be an integer, got {action_id!r}"
            )
        if not 0 <= action_id < self.n_actions:
            raise InvalidGymnasiumActionError(
                f"TPG action ID {action_id} is outside [0, {self.n_actions})"
            )
        environment_action = self._action_start + action_id
        contains = getattr(self.environment.action_space, "contains", None)
        if callable(contains):
            contains_action = cast(Callable[[object], object], contains)
            if not bool(contains_action(environment_action)):
                raise InvalidGymnasiumActionError(
                    f"mapped action {environment_action} is rejected by action_space"
                )
        return environment_action

    def close(self) -> None:
        """Close the owned environment exactly once."""

        if not self._closed:
            self.environment.close()
            self._closed = True
            self._needs_reset = True

    def __enter__(self) -> GymnasiumAdapter:
        self._require_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise GymnasiumEpisodeStateError("the environment adapter is closed")

    @staticmethod
    def _validate_action_space(space: object) -> tuple[int, int]:
        if getattr(space, "shape", None) != ():
            raise UnsupportedGymnasiumSpaceError(
                "action_space must be a scalar Discrete-like space"
            )
        try:
            n_actions = _index(cast(Any, space).n, "action_space.n")
        except AttributeError as error:
            raise UnsupportedGymnasiumSpaceError(
                "action_space must be a scalar Discrete-like space"
            ) from error
        if n_actions < 1:
            raise UnsupportedGymnasiumSpaceError(
                "action_space.n must be a positive integer"
            )
        start = _index(getattr(space, "start", 0), "action_space.start")
        return n_actions, start

    @staticmethod
    def _validate_observation_space(space: object) -> tuple[tuple[int, ...], int]:
        shape = getattr(space, "shape", None)
        if not isinstance(shape, tuple):
            raise UnsupportedGymnasiumSpaceError(
                "observation_space must have one fixed numeric shape; "
                "composite spaces are not supported"
            )
        untyped_shape = cast(tuple[object, ...], shape)
        dimensions = tuple(
            _index(value, f"observation_space.shape[{index}]")
            for index, value in enumerate(untyped_shape)
        )
        if any(value < 0 for value in dimensions):
            raise UnsupportedGymnasiumSpaceError(
                "observation_space dimensions must be non-negative"
            )
        input_size = math.prod(dimensions) if dimensions else 1
        if input_size < 1:
            raise UnsupportedGymnasiumSpaceError(
                "observation_space must contain at least one scalar"
            )
        dtype = getattr(space, "dtype", None)
        if getattr(dtype, "kind", None) not in {"i", "u", "f"}:
            raise UnsupportedGymnasiumSpaceError(
                "observation_space dtype must be integer or floating point"
            )
        return dimensions, input_size

    def _normalize_observation(self, value: object) -> tuple[float, ...]:
        numpy: Any = importlib.import_module("numpy")
        try:
            array: Any = numpy.asarray(value)
        except (TypeError, ValueError) as error:
            raise InvalidGymnasiumTransitionError(
                "observation cannot be converted to a numeric array"
            ) from error
        if tuple(array.shape) != self._observation_shape:
            raise InvalidGymnasiumTransitionError(
                f"observation shape {tuple(array.shape)!r} does not match "
                f"observation_space shape {self._observation_shape!r}"
            )
        if array.dtype.kind not in {"i", "u", "f"}:
            raise InvalidGymnasiumTransitionError(
                "observation values must be integer or floating point"
            )
        flattened: list[Any] = array.reshape(-1, order="C").tolist()
        normalized = tuple(float(item) for item in flattened)
        if len(normalized) != self.input_size or not all(
            math.isfinite(item) for item in normalized
        ):
            raise InvalidGymnasiumTransitionError(
                "observation must contain only finite numeric values"
            )
        return normalized

    @staticmethod
    def _normalize_reward(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidGymnasiumTransitionError(
                f"reward must be a finite real number, got {value!r}"
            )
        normalized = float(value)
        if not math.isfinite(normalized):
            raise InvalidGymnasiumTransitionError(
                f"reward must be finite, got {value!r}"
            )
        return normalized

    @staticmethod
    def _normalize_flag(value: object, name: str) -> bool:
        if not isinstance(value, bool):
            raise InvalidGymnasiumTransitionError(
                f"{name} must be a bool, got {value!r}"
            )
        return value

    @staticmethod
    def _normalize_info(value: object, name: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise InvalidGymnasiumTransitionError(f"{name} must be a mapping")
        untyped = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in untyped):
            raise InvalidGymnasiumTransitionError(f"{name} keys must all be strings")
        return MappingProxyType(cast(dict[str, object], dict(untyped)))

    @staticmethod
    def _require_result_tuple(
        value: object,
        length: int,
        operation: str,
    ) -> tuple[object, ...]:
        if not isinstance(value, tuple):
            raise InvalidGymnasiumTransitionError(
                f"{operation} must return a {length}-item tuple"
            )
        values = cast(tuple[object, ...], value)
        if len(values) != length:
            raise InvalidGymnasiumTransitionError(
                f"{operation} must return a {length}-item tuple"
            )
        return values


def make_gymnasium_environment(
    environment_id: str,
    **keyword_arguments: object,
) -> GymnasiumEnvironment:
    """Create an environment while keeping Gymnasium an optional dependency."""

    if not isinstance(environment_id, str) or not environment_id.strip():
        raise GymnasiumDependencyError("environment_id must be a non-empty string")
    try:
        gymnasium: Any = importlib.import_module("gymnasium")
    except ModuleNotFoundError as error:
        if error.name != "gymnasium":
            raise
        raise GymnasiumDependencyError(
            'Gymnasium is not installed; install pyTPG with the "gymnasium" extra'
        ) from error
    return cast(
        GymnasiumEnvironment,
        gymnasium.make(environment_id, **keyword_arguments),
    )


def _index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise UnsupportedGymnasiumSpaceError(f"{name} must be an integer")
    try:
        return operator.index(cast(Any, value))
    except TypeError as error:
        raise UnsupportedGymnasiumSpaceError(f"{name} must be an integer") from error


__all__ = [
    "GymnasiumAdapter",
    "GymnasiumEnvironment",
    "GymnasiumEnvironmentFactory",
    "GymnasiumReset",
    "GymnasiumStep",
    "make_gymnasium_environment",
]
