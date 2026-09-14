"""Dependency-free topology and session records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pytpg.runtime._model import GraphLike
from pytpg.runtime.trace import DetailedTraversalResult


@dataclass(frozen=True, slots=True)
class LearnerSnapshot:
    """A learner occurrence, identified by (team_id, learner_id)."""

    team_id: int
    learner_id: int
    program_id: int
    action_kind: Literal["atomic", "team_reference"]
    atomic_action_id: int | None
    referenced_team_id: int | None


@dataclass(frozen=True, slots=True)
class TeamSnapshot:
    team_id: int
    is_root: bool
    learners: tuple[LearnerSnapshot, ...]


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """Ordered topology only; no executable Python objects or observations."""

    teams: tuple[TeamSnapshot, ...]

    @classmethod
    def from_graph(cls, graph: GraphLike) -> GraphSnapshot:
        return cls(
            tuple(
                TeamSnapshot(
                    team.id,
                    team.id in graph.root_team_ids,
                    tuple(
                        LearnerSnapshot(
                            team.id,
                            item.id,
                            item.program.id,
                            item.action.kind,
                            item.action.action_id
                            if item.action.kind == "atomic"
                            else None,
                            item.action.team_id
                            if item.action.kind == "team_reference"
                            else None,
                        )
                        for item in team.learners
                    ),
                )
                for team in graph.teams
            )
        )


@dataclass(frozen=True, slots=True)
class TraceFrame:
    """An ordered decision with an independent environment step label.

    Metadata is held as canonical JSON so callers cannot mutate saved history.
    """

    frame_index: int
    step: int
    traversal: DetailedTraversalResult
    agent_id: str | None = None
    _metadata_json: str = "{}"

    @property
    def metadata(self) -> dict[str, Any]:
        from pytpg.serialization.io import load_json

        return load_json(self._metadata_json)


class TraceSession:
    """Append decisions for one fixed topology, without rendering dependencies."""

    def __init__(self, graph: GraphSnapshot) -> None:
        self._graph = graph
        self._frames: list[TraceFrame] = []

    @property
    def graph(self) -> GraphSnapshot:
        """The fixed immutable topology associated with this session."""
        return self._graph

    @classmethod
    def from_graph(cls, graph: GraphLike) -> TraceSession:
        return cls(GraphSnapshot.from_graph(graph))

    @property
    def frames(self) -> tuple[TraceFrame, ...]:
        return tuple(self._frames)

    def append(
        self,
        result: DetailedTraversalResult,
        *,
        step: int | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceFrame:
        from pytpg.serialization._validation import integer, mapping, string
        from pytpg.serialization.io import dump_json
        from pytpg.visualization.serialization import validate_metadata, validate_trace

        validate_trace(self.graph, result)
        index = len(self._frames)
        selected_step = index if step is None else integer(step, "step")
        if agent_id is not None:
            string(agent_id, "agent_id")
        payload = {} if metadata is None else mapping(metadata, "metadata")
        validate_metadata(payload)
        frame = TraceFrame(index, selected_step, result, agent_id, dump_json(payload))
        self._frames.append(frame)
        return frame

    def to_json(self, path: str | Path) -> None:
        from pytpg.serialization.io import write_atomic
        from pytpg.visualization.serialization import dumps_session

        write_atomic(path, dumps_session(self))

    @classmethod
    def from_json(cls, path: str | Path) -> TraceSession:
        from pytpg.serialization.io import read_text
        from pytpg.visualization.serialization import loads_session

        return loads_session(read_text(path))
