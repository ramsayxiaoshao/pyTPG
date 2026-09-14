"""Optional execution replay; imports never load Plotly or NetworkX."""

from pytpg.visualization.aggregate import ActivationCounts, aggregate
from pytpg.visualization.model import (
    GraphSnapshot,
    LearnerSnapshot,
    TeamSnapshot,
    TraceFrame,
    TraceSession,
)
from pytpg.visualization.render import (
    VisualizationDependencyError,
    build_figure,
    render_html,
)
from pytpg.visualization.serialization import (
    dumps_session,
    loads_session,
    snapshot_from_dict,
    snapshot_to_dict,
)

__all__ = [
    "ActivationCounts",
    "GraphSnapshot",
    "LearnerSnapshot",
    "TeamSnapshot",
    "TraceFrame",
    "TraceSession",
    "VisualizationDependencyError",
    "aggregate",
    "build_figure",
    "dumps_session",
    "loads_session",
    "render_html",
    "snapshot_from_dict",
    "snapshot_to_dict",
]
