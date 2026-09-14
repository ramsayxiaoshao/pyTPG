"""Explicit versioned JSON schemas and semantic validation for replay."""

from __future__ import annotations

from dataclasses import asdict
from typing import cast

from pytpg.runtime.trace import (
    DetailedTraversalResult,
    LearnerEvaluationTrace,
    TeamDecisionTrace,
)
from pytpg.serialization._validation import (
    boolean,
    exact_keys,
    finite_number,
    integer,
    mapping,
    sequence,
    string,
)
from pytpg.serialization.io import dump_json, load_json
from pytpg.visualization.model import (
    GraphSnapshot,
    LearnerSnapshot,
    TeamSnapshot,
    TraceSession,
)

SCHEMA_VERSION = 1


def validate_metadata(value: object) -> None:
    """Reject lossy coercions, non-string keys and non-finite JSON numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        finite_number(value, "metadata")
    elif isinstance(value, list):
        for item in cast(list[object], value):
            validate_metadata(item)
    elif isinstance(value, dict):
        for item in mapping(cast(dict[object, object], value), "metadata").values():
            validate_metadata(item)
    else:
        raise ValueError("metadata must contain only JSON-compatible values")


def snapshot_to_dict(graph: GraphSnapshot) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "teams": [
            {
                "team_id": t.team_id,
                "is_root": t.is_root,
                "learners": [asdict(item) for item in t.learners],
            }
            for t in graph.teams
        ],
    }


def snapshot_from_dict(value: object) -> GraphSnapshot:
    data = mapping(value, "topology")
    exact_keys(data, {"schema_version", "teams"}, "topology")
    _version(data["schema_version"])
    teams: list[TeamSnapshot] = []
    for raw in sequence(data["teams"], "teams"):
        team = mapping(raw, "team")
        exact_keys(team, {"team_id", "is_root", "learners"}, "team")
        team_id = integer(team["team_id"], "team_id")
        learners: list[LearnerSnapshot] = []
        for entry in sequence(team["learners"], "learners"):
            item = mapping(entry, "learner")
            exact_keys(
                item,
                {
                    "team_id",
                    "learner_id",
                    "program_id",
                    "action_kind",
                    "atomic_action_id",
                    "referenced_team_id",
                },
                "learner",
            )
            atomic, target = _destination(item)
            if integer(item["team_id"], "team_id") != team_id:
                raise ValueError("learner parent team does not match")
            learners.append(
                LearnerSnapshot(
                    team_id,
                    integer(item["learner_id"], "learner_id"),
                    integer(item["program_id"], "program_id"),
                    item["action_kind"],
                    atomic,
                    target,
                )
            )
        if not learners or len({v.learner_id for v in learners}) != len(learners):
            raise ValueError("team must have nonempty, unique learners")
        teams.append(
            TeamSnapshot(team_id, boolean(team["is_root"], "is_root"), tuple(learners))
        )
    ids = {t.team_id for t in teams}
    if not teams or len(ids) != len(teams) or not any(t.is_root for t in teams):
        raise ValueError("topology requires unique teams and at least one root")
    for team in teams:
        for item in team.learners:
            if (
                item.referenced_team_id is not None
                and item.referenced_team_id not in ids
            ):
                raise ValueError("unresolved team reference")
    return GraphSnapshot(tuple(teams))


def _version(value: object) -> None:
    if integer(value, "schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema version: {value}")


def _destination(item: dict[str, object]) -> tuple[int | None, int | None]:
    atomic = item["atomic_action_id"]
    target = item["referenced_team_id"]
    kind = item["action_kind"]
    if kind == "atomic" and target is None:
        return integer(atomic, "atomic_action_id"), None
    if kind == "team_reference" and atomic is None:
        return None, integer(target, "referenced_team_id")
    raise ValueError("action must have exactly one destination matching its kind")


def _trace_from_dict(value: object) -> DetailedTraversalResult:
    data = mapping(value, "traversal")
    exact_keys(data, {"root_team_id", "action_id", "team_decisions"}, "traversal")
    decisions: list[TeamDecisionTrace] = []
    for raw in sequence(data["team_decisions"], "team_decisions"):
        decision = mapping(raw, "decision")
        exact_keys(decision, {"team_id", "evaluations"}, "decision")
        evaluations: list[LearnerEvaluationTrace] = []
        for entry in sequence(decision["evaluations"], "evaluations"):
            item = mapping(entry, "evaluation")
            exact_keys(
                item,
                {
                    "learner_id",
                    "program_id",
                    "bid",
                    "evaluated",
                    "eligible",
                    "winner",
                    "action_kind",
                    "atomic_action_id",
                    "referenced_team_id",
                },
                "evaluation",
            )
            atomic, target = _destination(item)
            evaluations.append(
                LearnerEvaluationTrace(
                    integer(item["learner_id"], "learner_id"),
                    integer(item["program_id"], "program_id"),
                    None if item["bid"] is None else finite_number(item["bid"], "bid"),
                    boolean(item["evaluated"], "evaluated"),
                    boolean(item["eligible"], "eligible"),
                    boolean(item["winner"], "winner"),
                    "atomic" if atomic is not None else "team_reference",
                    atomic,
                    target,
                )
            )
        decisions.append(
            TeamDecisionTrace(
                integer(decision["team_id"], "team_id"), tuple(evaluations)
            )
        )
    return DetailedTraversalResult(
        integer(data["root_team_id"], "root_team_id"),
        integer(data["action_id"], "action_id"),
        tuple(decisions),
    )


def validate_trace(graph: GraphSnapshot, result: DetailedTraversalResult) -> None:
    """Check topology, exclusion, stable winners and path without executing code."""
    teams = {t.team_id: t for t in graph.teams}
    if result.root_team_id not in teams or not teams[result.root_team_id].is_root:
        raise ValueError("trace root is not a topology root")
    current = result.root_team_id
    visited: set[int] = set()
    if not result.team_decisions:
        raise ValueError("trace must contain decisions")
    for index, decision in enumerate(result.team_decisions):
        if decision.team_id != current or current in visited:
            raise ValueError("invalid traversal path")
        visited.add(current)
        team = teams[current]
        if len(team.learners) != len(decision.evaluations):
            raise ValueError("trace must include every learner")
        for learner, item in zip(team.learners, decision.evaluations, strict=True):
            if (
                learner.learner_id,
                learner.program_id,
                learner.action_kind,
                learner.atomic_action_id,
                learner.referenced_team_id,
            ) != (
                item.learner_id,
                item.program_id,
                item.action_kind,
                item.atomic_action_id,
                item.referenced_team_id,
            ):
                raise ValueError("trace learner does not match topology")
            eligible = (
                item.action_kind == "atomic" or item.referenced_team_id not in visited
            )
            if item.eligible != eligible or item.evaluated != eligible:
                raise ValueError("incorrect eligibility/evaluation state")
            if eligible:
                finite_number(item.bid, "bid")
            elif item.bid is not None or item.winner:
                raise ValueError("excluded learner cannot bid or win")
        eligible_items = [item for item in decision.evaluations if item.eligible]
        if not eligible_items:
            raise ValueError("decision has no eligible learners")
        winner = max(eligible_items, key=lambda item: float(cast(float, item.bid)))
        if [item for item in decision.evaluations if item.winner] != [winner]:
            raise ValueError("winner does not match stable maximum bid")
        if winner.atomic_action_id is not None:
            if (
                index != len(result.team_decisions) - 1
                or result.action_id != winner.atomic_action_id
            ):
                raise ValueError("invalid terminal action")
        else:
            if index == len(result.team_decisions) - 1:
                raise ValueError("trace must terminate at an atomic action")
            current = cast(int, winner.referenced_team_id)


def dumps_session(session: TraceSession) -> str:
    return dump_json(
        {
            "schema_version": SCHEMA_VERSION,
            "topology": snapshot_to_dict(session.graph),
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "step": frame.step,
                    "agent_id": frame.agent_id,
                    "metadata": frame.metadata,
                    "traversal": asdict(frame.traversal),
                }
                for frame in session.frames
            ],
        }
    )


def loads_session(text: str) -> TraceSession:
    data = mapping(load_json(text), "session")
    exact_keys(data, {"schema_version", "topology", "frames"}, "session")
    _version(data["schema_version"])
    session = TraceSession(snapshot_from_dict(data["topology"]))
    for frame_index, raw in enumerate(sequence(data["frames"], "frames")):
        item = mapping(raw, "frame")
        exact_keys(
            item, {"frame_index", "step", "agent_id", "metadata", "traversal"}, "frame"
        )
        if integer(item["frame_index"], "frame_index") != frame_index:
            raise ValueError("frame indices must be contiguous and ordered")
        session.append(
            _trace_from_dict(item["traversal"]),
            step=integer(item["step"], "step"),
            agent_id=None
            if item["agent_id"] is None
            else string(item["agent_id"], "agent_id"),
            metadata=mapping(item["metadata"], "metadata"),
        )
    return session
