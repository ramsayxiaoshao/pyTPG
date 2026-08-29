"""Versioned normalized graph JSON round-trip and rejection tests."""

import json
from pathlib import Path

import pytest

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
    RegisterOperand,
    Team,
    TeamID,
    TeamReference,
    TPGGraph,
)
from tpg.serialization import (
    GRAPH_FORMAT_VERSION,
    InvalidSerializedDataError,
    UnsupportedFormatVersionError,
    dumps_graph,
    graph_from_dict,
    graph_to_dict,
    load_graph,
    loads_graph,
    save_graph,
)


def shared_graph() -> TPGGraph:
    shared_program = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("add"),
                RegisterIndex(1),
                (InputOperand(InputIndex(0)), ConstantOperand(2.0)),
            ),
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (RegisterOperand(RegisterIndex(1)),),
            ),
        ),
    )
    shared = Learner(
        LearnerID(0), shared_program, AtomicAction(ActionID(1))
    )
    reference = Learner(
        LearnerID(1),
        Program(
            ProgramID(1),
            (
                Instruction(
                    OperatorName("identity"),
                    RegisterIndex(0),
                    (ConstantOperand(5.0),),
                ),
            ),
        ),
        TeamReference(TeamID(1)),
    )
    root = Team(TeamID(0), (reference, shared))
    target = Team(TeamID(1), (shared,))
    return TPGGraph((root, target), (root.id,))


def test_graph_json_round_trip_preserves_values_and_shared_identity() -> None:
    source = shared_graph()

    text = dumps_graph(source)
    restored = loads_graph(text)

    assert restored == source
    assert restored.teams[0].learners[1] is restored.teams[1].learners[0]
    document = json.loads(text)
    assert document["format_version"] == GRAPH_FORMAT_VERSION
    assert set(document["graph"]) == {
        "root_team_ids",
        "teams",
        "learners",
        "programs",
    }


def test_graph_file_save_is_atomic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "agent.tpg.json"

    save_graph(shared_graph(), path)

    assert load_graph(path) == shared_graph()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_graph_reader_rejects_unsupported_version() -> None:
    document = graph_to_dict(shared_graph())
    document["format_version"] = 99

    with pytest.raises(UnsupportedFormatVersionError, match="version 99"):
        graph_from_dict(document)


def test_graph_reader_rejects_unknown_fields_and_missing_memberships() -> None:
    unknown = graph_to_dict(shared_graph())
    unknown["extra"] = True
    with pytest.raises(InvalidSerializedDataError, match="fields"):
        graph_from_dict(unknown)

    dangling = graph_to_dict(shared_graph())
    graph = dangling["graph"]  # type: ignore[assignment]
    graph["teams"][0]["learner_ids"][0] = 999  # type: ignore[index]
    with pytest.raises(InvalidSerializedDataError, match="missing learner"):
        graph_from_dict(dangling)


def test_graph_reader_rejects_invalid_json_and_unused_tables() -> None:
    with pytest.raises(InvalidSerializedDataError, match="invalid JSON"):
        loads_graph("{not-json")

    document = graph_to_dict(shared_graph())
    graph = document["graph"]  # type: ignore[assignment]
    graph["learners"].append(  # type: ignore[index,union-attr]
        {"id": 99, "program_id": 1, "action": {"kind": "atomic", "action_id": 0}}
    )
    with pytest.raises(InvalidSerializedDataError, match="unused"):
        graph_from_dict(document)
