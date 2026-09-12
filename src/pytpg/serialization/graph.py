"""Version-1 normalized JSON serialization for immutable TPG graphs."""

from __future__ import annotations

from pathlib import Path

from pytpg.core import (
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
from pytpg.serialization._validation import (
    exact_keys,
    finite_number,
    integer,
    mapping,
    sequence,
    string,
)
from pytpg.serialization.errors import (
    InvalidSerializedDataError,
    UnsupportedFormatVersionError,
)
from pytpg.serialization.io import dump_json, load_json, read_text, write_atomic

GRAPH_FORMAT = "pytpg-graph"
GRAPH_FORMAT_VERSION = 1


def graph_to_data(graph: TPGGraph) -> dict[str, object]:
    """Return normalized graph tables without an outer format envelope."""

    learners: dict[int, Learner] = {}
    programs: dict[int, Program] = {}
    for team in graph.teams:
        for learner in team.learners:
            learners.setdefault(int(learner.id), learner)
            programs.setdefault(int(learner.program.id), learner.program)
    return {
        "root_team_ids": [int(team_id) for team_id in graph.root_team_ids],
        "teams": [
            {
                "id": int(team.id),
                "learner_ids": [int(learner.id) for learner in team.learners],
            }
            for team in graph.teams
        ],
        "learners": [
            {
                "id": int(learner.id),
                "program_id": int(learner.program.id),
                "action": _action_to_data(learner),
            }
            for learner in learners.values()
        ],
        "programs": [
            {
                "id": int(program.id),
                "instructions": [
                    _instruction_to_data(instruction)
                    for instruction in program.instructions
                ],
            }
            for program in programs.values()
        ],
    }


def graph_from_data(value: object) -> TPGGraph:
    """Rebuild and context-free validate one normalized graph."""

    data = mapping(value, "graph")
    exact_keys(data, {"root_team_ids", "teams", "learners", "programs"}, "graph")
    try:
        programs = _programs_from_data(data["programs"])
        learners = _learners_from_data(data["learners"], programs)
        teams = _teams_from_data(data["teams"], learners)
        used_learners = {
            int(learner.id) for team in teams for learner in team.learners
        }
        if used_learners != set(learners):
            raise InvalidSerializedDataError(
                "graph learner table contains unused entries"
            )
        used_programs = {int(learner.program.id) for learner in learners.values()}
        if used_programs != set(programs):
            raise InvalidSerializedDataError(
                "graph program table contains unused entries"
            )
        roots = tuple(
            TeamID(integer(item, f"graph.root_team_ids[{index}]"))
            for index, item in enumerate(
                sequence(data["root_team_ids"], "graph.root_team_ids")
            )
        )
        return TPGGraph(teams, roots)
    except (TypeError, ValueError) as error:
        if isinstance(error, InvalidSerializedDataError):
            raise
        raise InvalidSerializedDataError(f"invalid graph data: {error}") from error


def graph_to_dict(graph: TPGGraph) -> dict[str, object]:
    """Return the complete versioned graph document."""

    return {
        "format": GRAPH_FORMAT,
        "format_version": GRAPH_FORMAT_VERSION,
        "graph": graph_to_data(graph),
    }


def graph_from_dict(value: object) -> TPGGraph:
    """Read an exact supported graph document."""

    document = mapping(value, "document")
    exact_keys(document, {"format", "format_version", "graph"}, "document")
    if string(document["format"], "document.format") != GRAPH_FORMAT:
        raise InvalidSerializedDataError("document is not a pyTPG graph")
    version = integer(document["format_version"], "document.format_version")
    if version != GRAPH_FORMAT_VERSION:
        raise UnsupportedFormatVersionError(
            f"unsupported graph format version {version}"
        )
    return graph_from_data(document["graph"])


def dumps_graph(graph: TPGGraph) -> str:
    return dump_json(graph_to_dict(graph))


def loads_graph(text: str) -> TPGGraph:
    return graph_from_dict(load_json(text))


def save_graph(graph: TPGGraph, path: str | Path) -> None:
    write_atomic(path, dumps_graph(graph))


def load_graph(path: str | Path) -> TPGGraph:
    return loads_graph(read_text(path))


def _action_to_data(learner: Learner) -> dict[str, object]:
    if isinstance(learner.action, AtomicAction):
        return {"kind": "atomic", "action_id": int(learner.action.action_id)}
    return {"kind": "team_reference", "team_id": int(learner.action.team_id)}


def _instruction_to_data(instruction: Instruction) -> dict[str, object]:
    operands: list[dict[str, object]] = []
    for operand in instruction.operands:
        if isinstance(operand, InputOperand):
            operands.append({"kind": "input", "index": int(operand.index)})
        elif isinstance(operand, RegisterOperand):
            operands.append({"kind": "register", "index": int(operand.index)})
        else:
            operands.append({"kind": "constant", "value": float(operand.value)})
    return {
        "operator": str(instruction.operator),
        "destination": int(instruction.destination),
        "operands": operands,
    }


def _programs_from_data(value: object) -> dict[int, Program]:
    programs: dict[int, Program] = {}
    for index, raw in enumerate(sequence(value, "graph.programs")):
        location = f"graph.programs[{index}]"
        item = mapping(raw, location)
        exact_keys(item, {"id", "instructions"}, location)
        program_id = integer(item["id"], f"{location}.id")
        if program_id in programs:
            raise InvalidSerializedDataError(f"duplicate program ID {program_id}")
        instructions = tuple(
            _instruction_from_data(instruction, f"{location}.instructions[{position}]")
            for position, instruction in enumerate(
                sequence(item["instructions"], f"{location}.instructions")
            )
        )
        programs[program_id] = Program(ProgramID(program_id), instructions)
    return programs


def _instruction_from_data(value: object, location: str) -> Instruction:
    item = mapping(value, location)
    exact_keys(item, {"operator", "destination", "operands"}, location)
    operands = tuple(
        _operand_from_data(operand, f"{location}.operands[{index}]")
        for index, operand in enumerate(
            sequence(item["operands"], f"{location}.operands")
        )
    )
    return Instruction(
        OperatorName(string(item["operator"], f"{location}.operator")),
        RegisterIndex(integer(item["destination"], f"{location}.destination")),
        operands,
    )


def _operand_from_data(
    value: object,
    location: str,
) -> InputOperand | RegisterOperand | ConstantOperand:
    item = mapping(value, location)
    kind = string(item.get("kind"), f"{location}.kind")
    if kind == "input":
        exact_keys(item, {"kind", "index"}, location)
        return InputOperand(InputIndex(integer(item["index"], f"{location}.index")))
    if kind == "register":
        exact_keys(item, {"kind", "index"}, location)
        return RegisterOperand(
            RegisterIndex(integer(item["index"], f"{location}.index"))
        )
    if kind == "constant":
        exact_keys(item, {"kind", "value"}, location)
        return ConstantOperand(finite_number(item["value"], f"{location}.value"))
    raise InvalidSerializedDataError(f"{location}.kind is unsupported")


def _learners_from_data(
    value: object,
    programs: dict[int, Program],
) -> dict[int, Learner]:
    learners: dict[int, Learner] = {}
    for index, raw in enumerate(sequence(value, "graph.learners")):
        location = f"graph.learners[{index}]"
        item = mapping(raw, location)
        exact_keys(item, {"id", "program_id", "action"}, location)
        learner_id = integer(item["id"], f"{location}.id")
        if learner_id in learners:
            raise InvalidSerializedDataError(f"duplicate learner ID {learner_id}")
        program_id = integer(item["program_id"], f"{location}.program_id")
        if program_id not in programs:
            raise InvalidSerializedDataError(
                f"learner {learner_id} references missing program {program_id}"
            )
        learners[learner_id] = Learner(
            LearnerID(learner_id),
            programs[program_id],
            _action_from_data(item["action"], f"{location}.action"),
        )
    return learners


def _action_from_data(value: object, location: str) -> AtomicAction | TeamReference:
    item = mapping(value, location)
    kind = string(item.get("kind"), f"{location}.kind")
    if kind == "atomic":
        exact_keys(item, {"kind", "action_id"}, location)
        return AtomicAction(
            ActionID(integer(item["action_id"], f"{location}.action_id"))
        )
    if kind == "team_reference":
        exact_keys(item, {"kind", "team_id"}, location)
        return TeamReference(
            TeamID(integer(item["team_id"], f"{location}.team_id"))
        )
    raise InvalidSerializedDataError(f"{location}.kind is unsupported")


def _teams_from_data(
    value: object,
    learners: dict[int, Learner],
) -> tuple[Team, ...]:
    teams: list[Team] = []
    team_ids: set[int] = set()
    for index, raw in enumerate(sequence(value, "graph.teams")):
        location = f"graph.teams[{index}]"
        item = mapping(raw, location)
        exact_keys(item, {"id", "learner_ids"}, location)
        team_id = integer(item["id"], f"{location}.id")
        if team_id in team_ids:
            raise InvalidSerializedDataError(f"duplicate team ID {team_id}")
        team_ids.add(team_id)
        memberships: list[Learner] = []
        for position, raw_learner_id in enumerate(
            sequence(item["learner_ids"], f"{location}.learner_ids")
        ):
            learner_id = integer(
                raw_learner_id, f"{location}.learner_ids[{position}]"
            )
            if learner_id not in learners:
                raise InvalidSerializedDataError(
                    f"team {team_id} references missing learner {learner_id}"
                )
            memberships.append(learners[learner_id])
        teams.append(Team(TeamID(team_id), tuple(memberships)))
    return tuple(teams)


__all__ = [
    "GRAPH_FORMAT",
    "GRAPH_FORMAT_VERSION",
    "dumps_graph",
    "graph_from_data",
    "graph_from_dict",
    "graph_to_data",
    "graph_to_dict",
    "load_graph",
    "loads_graph",
    "save_graph",
]
