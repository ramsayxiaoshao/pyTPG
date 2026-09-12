"""Shared validation helpers for deterministic multi-agent composition."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pytpg.multiagent.errors import (
    MultiAgentConfigurationError,
    MultiAgentObservationError,
)
from pytpg.multiagent.model import AgentID, AgentSpec


def validate_specs(specs: tuple[AgentSpec, ...]) -> tuple[AgentSpec, ...]:
    if not isinstance(specs, tuple):
        raise MultiAgentConfigurationError("agent specs must be a tuple")
    if not specs:
        raise MultiAgentConfigurationError("at least one agent spec is required")
    if not all(isinstance(spec, AgentSpec) for spec in specs):
        raise MultiAgentConfigurationError("agent specs contain an unsupported value")

    seen: set[AgentID] = set()
    for spec in specs:
        if spec.agent_id in seen:
            raise MultiAgentConfigurationError(
                f"agent roster repeats agent ID {spec.agent_id!r}"
            )
        seen.add(spec.agent_id)
    return specs


def ordered_observations(
    specs: tuple[AgentSpec, ...],
    observations: Mapping[AgentID, Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(observations, Mapping):
        raise MultiAgentObservationError("observations must be a mapping by agent ID")

    expected = {spec.agent_id for spec in specs}
    actual = set(observations)
    if actual != expected:
        missing = tuple(spec.agent_id for spec in specs if spec.agent_id not in actual)
        unexpected = tuple(
            sorted(
                (agent_id for agent_id in actual if agent_id not in expected),
                key=repr,
            )
        )
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unexpected:
            details.append(f"unexpected={unexpected!r}")
        raise MultiAgentObservationError(
            "observations must contain exactly the configured agent IDs ("
            + ", ".join(details)
            + ")"
        )

    return tuple(
        _finite_observation(observations[spec.agent_id], spec) for spec in specs
    )


def _finite_observation(
    observation: Sequence[float], spec: AgentSpec
) -> tuple[float, ...]:
    if isinstance(observation, (str, bytes)):
        raise MultiAgentObservationError(
            f"observation for agent {spec.agent_id!r} must be a numeric sequence"
        )
    try:
        values = tuple(observation)
    except TypeError as error:
        raise MultiAgentObservationError(
            f"observation for agent {spec.agent_id!r} must be a sequence"
        ) from error
    if len(values) != spec.observation_size:
        raise MultiAgentObservationError(
            f"observation for agent {spec.agent_id!r} has length {len(values)}; "
            f"expected {spec.observation_size}"
        )

    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MultiAgentObservationError(
                f"observation for agent {spec.agent_id!r} at index {index} "
                "must be a finite real number"
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise MultiAgentObservationError(
                f"observation for agent {spec.agent_id!r} at index {index} "
                "must be finite"
            )
        normalized.append(converted)
    return tuple(normalized)


__all__ = ["ordered_observations", "validate_specs"]
