"""Deterministic concentric hierarchy and exact node silhouettes."""

from __future__ import annotations

import math
from typing import Any


def radial_layout(
    topology: Any, root: str
) -> tuple[dict[str, tuple[float, float]], int]:
    """Place a BFS spanning tree on rings; retain all cross/back edges separately."""
    children: dict[str, list[str]] = {node: [] for node in topology.nodes}
    depths = {root: 0}
    order = [root]
    for node in order:
        for target in topology.successors(node):
            if target not in depths:
                depths[target] = depths[node] + 1
                children[node].append(target)
                order.append(target)
    # Unreachable components get their own outer sectors, without hiding nodes.
    for node in topology.nodes:
        if node not in depths:
            depths[node] = 1
            children[root].append(node)
            order.append(node)
            pending = [node]
            for parent in pending:
                for target in topology.successors(parent):
                    if target not in depths:
                        depths[target] = depths[parent] + 1
                        children[parent].append(target)
                        order.append(target)
                        pending.append(target)
    weights: dict[str, int] = {}
    for node in reversed(order):
        weights[node] = max(1, sum(weights[c] for c in children[node]))
    positions = {root: (0.0, 0.0)}
    sectors = {root: (-math.pi / 2, 3 * math.pi / 2)}
    for node in order:
        start, end = sectors[node]
        cursor = start
        for child in children[node]:
            span = (end - start) * weights[child] / weights[node]
            angle = cursor + span / 2
            sectors[child] = (cursor, cursor + span)
            positions[child] = (
                depths[child] * math.cos(angle),
                depths[child] * math.sin(angle),
            )
            cursor += span
    return positions, max(depths.values())


def node_shape(
    kind: str, x: float, y: float, color: str, opacity: float, border: str, width: int
) -> dict[str, object]:
    """Pixel-sized shapes retain true circles, ellipses and rounded rectangles."""
    shape: dict[str, object] = {
        "xref": "x",
        "yref": "y",
        "xanchor": x,
        "yanchor": y,
        "xsizemode": "pixel",
        "ysizemode": "pixel",
        "layer": "above",
        "fillcolor": color,
        "opacity": opacity,
        "line": {"color": border, "width": width},
    }
    if kind == "team":
        shape.update(
            type="path",
            path=(
                "M -30,-19 L 30,-19 Q 40,-19 40,-9 L 40,9 Q 40,19 30,19 "
                "L -30,19 Q -40,19 -40,9 L -40,-9 Q -40,-19 -30,-19 Z"
            ),
        )
    else:
        rx, ry = (36, 20) if kind == "action" else (23, 23)
        shape.update(type="circle", x0=-rx, x1=rx, y0=-ry, y1=ry)
    return shape
