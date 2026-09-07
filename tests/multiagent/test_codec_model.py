"""Tests for multi-agent value objects and joint-action encoding."""

import pytest

from tpg.multiagent import (
    AgentID,
    AgentSpec,
    JointActionCodec,
    MultiAgentActionError,
    MultiAgentConfigurationError,
)


def specs() -> tuple[AgentSpec, ...]:
    return (
        AgentSpec(AgentID("scout"), observation_size=1, action_count=2),
        AgentSpec(AgentID("arm"), observation_size=2, action_count=3),
    )


def test_mixed_radix_codec_round_trips_heterogeneous_actions() -> None:
    codec = JointActionCodec(specs())

    encoded = codec.encode({AgentID("arm"): 2, AgentID("scout"): 1})
    decoded = codec.decode(encoded)

    assert codec.action_count == 6
    assert encoded == 5
    assert tuple((item.agent_id, item.action_id) for item in decoded) == (
        (AgentID("scout"), 1),
        (AgentID("arm"), 2),
    )


@pytest.mark.parametrize("joint_action", [-1, 6, True, 1.5])
def test_joint_action_decode_rejects_values_outside_product_space(
    joint_action: object,
) -> None:
    with pytest.raises(MultiAgentActionError, match="outside"):
        JointActionCodec(specs()).decode(joint_action)  # type: ignore[arg-type]


def test_joint_action_encode_requires_exact_agent_set_and_bounds() -> None:
    codec = JointActionCodec(specs())

    with pytest.raises(MultiAgentActionError, match="exactly"):
        codec.encode({AgentID("scout"): 0})
    with pytest.raises(MultiAgentActionError, match="arm.*outside"):
        codec.encode({AgentID("scout"): 0, AgentID("arm"): 3})


@pytest.mark.parametrize(
    ("agent_id", "observation_size", "action_count"),
    [
        (AgentID(""), 1, 1),
        (AgentID(" agent"), 1, 1),
        (AgentID("agent"), -1, 1),
        (AgentID("agent"), 1, 0),
        (AgentID("agent"), True, 1),
    ],
)
def test_agent_spec_rejects_ambiguous_or_invalid_dimensions(
    agent_id: AgentID, observation_size: int, action_count: int
) -> None:
    with pytest.raises(MultiAgentConfigurationError):
        AgentSpec(agent_id, observation_size, action_count)


def test_roster_requires_nonempty_unique_tuple() -> None:
    spec = AgentSpec(AgentID("agent"), 1, 2)

    with pytest.raises(MultiAgentConfigurationError, match="tuple"):
        JointActionCodec([spec])  # type: ignore[arg-type]
    with pytest.raises(MultiAgentConfigurationError, match="at least one"):
        JointActionCodec(())
    with pytest.raises(MultiAgentConfigurationError, match="repeats"):
        JointActionCodec((spec, spec))
