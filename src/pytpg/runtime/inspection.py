"""Deterministic structural summaries for TPG research inspection."""

from dataclasses import dataclass

from pytpg.runtime._model import GraphLike, LearnerLike, ProgramLike, TeamLike


@dataclass(frozen=True, slots=True)
class GraphSummary:
    """Basic graph statistics with counts based on stable entity IDs."""

    root_team_count: int
    team_count: int
    reachable_team_count: int
    orphan_team_count: int
    cyclic_team_count: int
    learner_count: int
    program_count: int
    instruction_count: int
    mean_program_length: float
    atomic_action_learner_count: int
    team_reference_learner_count: int


def _team_index(graph: GraphLike) -> dict[int, TeamLike]:
    return {team.id: team for team in graph.teams}


def _adjacency(graph: GraphLike) -> dict[int, tuple[int, ...]]:
    teams_by_id = _team_index(graph)
    return {
        team.id: tuple(
            learner.action.team_id
            for learner in team.learners
            if learner.action.kind == "team_reference"
            and learner.action.team_id in teams_by_id
        )
        for team in graph.teams
    }


def reachable_team_ids(graph: GraphLike) -> frozenset[int]:
    """Return every team reachable from a declared, resolving root."""

    adjacency = _adjacency(graph)
    reachable: set[int] = set()
    pending = [root for root in graph.root_team_ids if root in adjacency]
    while pending:
        team_id = pending.pop()
        if team_id in reachable:
            continue
        reachable.add(team_id)
        pending.extend(adjacency[team_id])
    return frozenset(reachable)


def cyclic_team_ids(graph: GraphLike) -> frozenset[int]:
    """Return teams belonging to at least one directed reference cycle."""

    adjacency = _adjacency(graph)
    cyclic: set[int] = set()
    for start in adjacency:
        pending = list(adjacency[start])
        explored: set[int] = set()
        while pending:
            team_id = pending.pop()
            if team_id == start:
                cyclic.add(start)
                break
            if team_id not in explored:
                explored.add(team_id)
                pending.extend(adjacency.get(team_id, ()))
    return frozenset(cyclic)


def summarize_graph(graph: GraphLike) -> GraphSummary:
    """Compute ID-deduplicated statistics without executing the graph."""

    learners: dict[int, LearnerLike] = {}
    programs: dict[int, ProgramLike] = {}
    for team in graph.teams:
        for learner in team.learners:
            learners.setdefault(learner.id, learner)
            programs.setdefault(learner.program.id, learner.program)

    reachable = reachable_team_ids(graph)
    cyclic = cyclic_team_ids(graph)
    instruction_count = sum(len(program.instructions) for program in programs.values())
    program_count = len(programs)
    return GraphSummary(
        root_team_count=len(graph.root_team_ids),
        team_count=len(graph.teams),
        reachable_team_count=len(reachable),
        orphan_team_count=len(graph.teams) - len(reachable),
        cyclic_team_count=len(cyclic),
        learner_count=len(learners),
        program_count=program_count,
        instruction_count=instruction_count,
        mean_program_length=(
            instruction_count / program_count if program_count else 0.0
        ),
        atomic_action_learner_count=sum(
            learner.action.kind == "atomic" for learner in learners.values()
        ),
        team_reference_learner_count=sum(
            learner.action.kind == "team_reference" for learner in learners.values()
        ),
    )


__all__ = [
    "GraphSummary",
    "cyclic_team_ids",
    "reachable_team_ids",
    "summarize_graph",
]
