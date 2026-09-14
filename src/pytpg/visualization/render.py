"""Lazy NetworkX layout and offline Plotly decision replay."""

from __future__ import annotations

import importlib
import json
import math
from html import escape
from pathlib import Path
from typing import Any

from pytpg.visualization.aggregate import aggregate
from pytpg.visualization.layout import node_shape, radial_layout
from pytpg.visualization.model import TraceFrame, TraceSession


class VisualizationDependencyError(ImportError):
    """Rendering was requested without the optional dependencies."""


def _dependencies() -> tuple[Any, Any]:
    # Dynamic imports keep third-party types and imports outside the data layer.
    try:
        return (
            importlib.import_module("networkx"),
            importlib.import_module("plotly.graph_objects"),
        )
    except ImportError as error:
        raise VisualizationDependencyError(
            'Visualization support requires: pip install "pytpg[visualization]"'
        ) from error


def _title(frame: TraceFrame | None) -> str:
    if frame is None:
        return "TPG execution replay · No recorded decisions"
    trace = frame.traversal
    path = " → ".join(
        [*(f"T{i}" for i in trace.visited_team_ids), f"A{trace.action_id}"]
    )
    agent = "" if frame.agent_id is None else f" · Agent: {escape(frame.agent_id)}"
    metadata = json.dumps(frame.metadata, sort_keys=True, ensure_ascii=False)
    compact = escape(metadata[:160] + ("…" if len(metadata) > 160 else ""))
    return (
        f"Step {frame.step}{agent} · Final action: A{trace.action_id}"
        f"<br><sup>Traversal: {path}"
        + (f" · {compact}" if frame.metadata else "")
        + "</sup>"
    )


def build_figure(session: TraceSession, *, layout_seed: int = 42) -> Any:
    """Build one fixed layout with a complete visual state for every decision."""
    nx, go = _dependencies()
    topology = nx.DiGraph()
    labels: dict[str, str] = {}
    kinds: dict[str, str] = {}
    base_hover: dict[str, str] = {}
    edges: list[tuple[str, str, tuple[int, int]]] = []
    counts = aggregate(session)
    for team in session.graph.teams:
        node = f"t:{team.team_id}"
        topology.add_node(node)
        labels[node], kinds[node] = f"T{team.team_id}", "team"
        base_hover[node] = (
            f"Team ID: {team.team_id}<br>Root: {team.is_root}"
            f"<br>Learners: {len(team.learners)}"
            f"<br>Session visits: {counts.team_visits[team.team_id]}"
        )
        for item in team.learners:
            key = team.team_id, item.learner_id
            learner = f"l:{team.team_id}:{item.learner_id}"
            destination = (
                f"a:{item.atomic_action_id}"
                if item.action_kind == "atomic"
                else f"t:{item.referenced_team_id}"
            )
            topology.add_edge(node, learner)
            topology.add_edge(learner, destination)
            edges.extend(((node, learner, key), (learner, destination, key)))
            labels[learner] = f"P{item.program_id} / L{item.learner_id}"
            kinds[learner] = "learner"
            base_hover[learner] = (
                f"Team ID: {team.team_id}<br>Learner ID: {item.learner_id}"
                f"<br>Program ID: {item.program_id}<br>Action kind: {item.action_kind}"
                f"<br>Atomic action ID: {item.atomic_action_id}"
                f"<br>Referenced Team ID: {item.referenced_team_id}"
                f"<br>Session occurrence evaluations: {counts.learner_evaluations[key]}"
                f"<br>Session occurrence wins: {counts.learner_wins[key]}"
                "<br>Session program evaluations: "
                f"{counts.program_evaluations[item.program_id]}"
                f"<br>Session program wins: {counts.program_wins[item.program_id]}"
            )
            if item.atomic_action_id is not None:
                labels[destination], kinds[destination] = (
                    f"A{item.atomic_action_id}",
                    "action",
                )
                base_hover[destination] = (
                    f"Action ID: {item.atomic_action_id}"
                    "<br>Session selections: "
                    f"{counts.atomic_actions[item.atomic_action_id]}"
                )
    # Kept for API compatibility: the hierarchy has no random component.
    del layout_seed
    roots = [f"t:{team.team_id}" for team in session.graph.teams if team.is_root]
    root = "t:0" if "t:0" in roots else roots[0]
    positions, depth = radial_layout(topology, root)
    extent = max(1.5, depth + 0.6)
    # Increase canvas spacing for crowded rings instead of packing icons together.
    closest = 1.0
    for level in range(1, depth + 1):
        angles = sorted(
            math.atan2(y, x)
            for x, y in positions.values()
            if math.isclose(math.hypot(x, y), level)
        )
        if len(angles) > 1:
            gaps = [b - a for a, b in zip(angles, angles[1:], strict=False)]
            gaps.append(angles[0] + 2 * math.pi - angles[-1])
            closest = min(closest, 2 * level * math.sin(min(gaps) / 2))
    spacing = max(115, 110 / max(closest, 0.001))
    rings = [
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=-level,
            y0=-level,
            x1=level,
            y1=level,
            line=dict(color="#e2e8f0", width=1, dash="dot"),
            layer="below",
        )
        for level in range(1, depth + 1)
    ]
    frame_shapes: list[list[Any]] = []
    nodes = list(labels)

    def frame_data(frame: TraceFrame | None) -> list[Any]:
        visited: set[int] = (
            set() if frame is None else set(frame.traversal.visited_team_ids)
        )
        evaluations = (
            {}
            if frame is None
            else {
                (decision.team_id, item.learner_id): item
                for decision in frame.traversal.team_decisions
                for item in decision.evaluations
            }
        )
        traces: list[Any] = []
        for source, target, key in edges:
            item = evaluations.get(key)
            selected = item is not None and item.winner
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            # A point near the destination makes edge counts hoverable.
            traces.append(
                go.Scatter(
                    x=[float(x0), float(x0 * 0.25 + x1 * 0.75), float(x1)],
                    y=[float(y0), float(y0 * 0.25 + y1 * 0.75), float(y1)],
                    mode="lines+markers",
                    marker={"size": 5, "opacity": 0.35},
                    line={
                        "width": 4 if selected else 1,
                        "color": "#f59e0b" if selected else "#b8c2d0",
                    },
                    opacity=1 if selected else 0.35,
                    text=(
                        f"{labels[source]} → {labels[target]}"
                        f"<br>Session traversals: {counts.edge_traversals[key]}"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                )
            )
        shapes: list[Any] = list(rings)
        colors: list[str] = []
        sizes: list[int] = []
        widths: list[int] = []
        opacity: list[float] = []
        symbols: list[str] = []
        hover: list[str] = []
        for node in nodes:
            kind = kinds[node]
            active = False
            winning = False
            excluded = False
            detail = ""
            if kind == "team":
                active = int(node.split(":")[1]) in visited
                detail = f"<br>Visited this decision: {active}"
            elif kind == "action":
                active = (
                    frame is not None
                    and int(node.split(":")[1]) == frame.traversal.action_id
                )
                detail = f"<br>Final action this decision: {active}"
            else:
                _, team_id, learner_id = node.split(":")
                item = evaluations.get((int(team_id), int(learner_id)))
                if item is not None:
                    active, winning, excluded = (
                        item.evaluated,
                        item.winner,
                        not item.eligible,
                    )
                    detail = (
                        f"<br>eligible: {item.eligible}<br>evaluated: {item.evaluated}"
                        f"<br>bid: {item.bid}<br>winner: {item.winner}"
                    )
                else:
                    detail = (
                        "<br>eligible: unknown (team not visited)"
                        "<br>evaluated: False<br>bid: None<br>winner: False"
                    )
            colors.append(
                {"team": "#2878A5", "action": "#8C3A5B", "learner": "#78B7D0"}[kind]
            )
            sizes.append(64)
            widths.append(4 if winning else 3 if active else 1)
            opacity.append(1 if active else 0.65 if excluded else 0.4)
            symbols.append("circle")
            x, y = positions[node]
            shapes.append(
                node_shape(
                    kind,
                    x,
                    y,
                    colors[-1],
                    opacity[-1],
                    "#f59e0b" if winning else "#172f46",
                    widths[-1],
                )
            )
            hover.append(base_hover[node] + detail)
        traces.append(
            go.Scatter(
                x=[float(positions[n][0]) for n in nodes],
                y=[float(positions[n][1]) for n in nodes],
                mode="markers+text",
                text=[labels[n] for n in nodes],
                textposition="top center",
                hovertext=hover,
                hoverinfo="text",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": sizes,
                    "symbol": symbols,
                    "opacity": opacity,
                    "line": {"width": 0},
                },
                showlegend=False,
            )
        )
        frame_shapes.append(shapes)
        return traces

    frames = session.frames
    figure = go.Figure(data=frame_data(frames[0] if frames else None))
    figure.frames = [
        go.Frame(
            name=str(frame.frame_index),
            data=frame_data(frame),
            layout={"title": {"text": _title(frame)}, "shapes": frame_shapes[-1]},
        )
        for frame in frames
    ]
    figure.update_layout(
        title={"text": _title(frames[0] if frames else None), "x": 0.03},
        template="plotly_white",
        height=max(900, int(extent * 2 * spacing + 260)),
        width=max(1000, int(extent * 2 * spacing + 100)),
        shapes=frame_shapes[0],
        margin={"t": 115, "b": 145, "l": 35, "r": 35},
        xaxis={"visible": False, "range": [-extent, extent], "fixedrange": False},
        yaxis={"visible": False, "range": [-extent, extent], "scaleanchor": "x"},
        uirevision="fixed-topology",
        hovermode="closest",
        annotations=[
            {
                "text": "Blue rounded rectangle: Team · Light blue circle: Program"
                " · Burgundy ellipse: Action"
                "<br>Rings: traversal depth · Dim: inactive · Thick border: active"
                " · Gold border/edge: winner/path"
                "<br>Hover for CURRENT DECISION and WHOLE SESSION counts",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.08,
                "showarrow": False,
                "align": "left",
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.22,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 650, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0.2,
                "len": 0.8,
                "y": -0.18,
                "currentvalue": {"prefix": "Decision: "},
                "steps": [
                    {
                        "label": f"{f.frame_index} (step {f.step})",
                        "method": "animate",
                        "args": [
                            [str(f.frame_index)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for f in frames
                ],
            }
        ],
    )
    return figure


def render_html(
    session: TraceSession, path: str | Path, *, layout_seed: int = 42
) -> Path:
    """Write standalone HTML with embedded Plotly JS; never open a browser."""
    figure = build_figure(session, layout_seed=layout_seed)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(target), include_plotlyjs=True, full_html=True, auto_play=False
    )
    return target
