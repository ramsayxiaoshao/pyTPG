"""Execution, persistence, optional dependency and replay regressions."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from pytpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
    Instruction,
    Learner,
    LearnerID,
    OperatorName,
    Program,
    ProgramID,
    RegisterIndex,
    Team,
    TeamID,
    TeamReference,
    TPGGraph,
)
from pytpg.runtime import GraphRuntime, ProgramExecutor, RuntimeConfig, TraversalStep
from pytpg.visualization import (
    TraceSession,
    VisualizationDependencyError,
    aggregate,
    build_figure,
    dumps_session,
    loads_session,
    render_html,
    snapshot_from_dict,
    snapshot_to_dict,
)
from pytpg.visualization.cli import main


def policy() -> TPGGraph:
    def learner(identifier, bid, action):
        return Learner(
            LearnerID(identifier),
            Program(
                ProgramID(identifier + 10),
                (
                    Instruction(
                        OperatorName("identity"),
                        RegisterIndex(0),
                        (ConstantOperand(bid),),
                    ),
                ),
            ),
            action,
        )

    shared = learner(9, -1.0, AtomicAction(ActionID(4)))
    return TPGGraph(
        (
            Team(TeamID(0), (learner(0, 2.0, TeamReference(TeamID(1))), shared)),
            Team(
                TeamID(1),
                (
                    learner(1, 100.0, TeamReference(TeamID(0))),
                    shared,
                    learner(2, -1.0, AtomicAction(ActionID(8))),
                ),
            ),
        ),
        (TeamID(0),),
    )


def session() -> TraceSession:
    graph = policy()
    result = GraphRuntime(RuntimeConfig(0, 1)).traverse_detailed(graph, ())
    output = TraceSession.from_graph(graph)
    output.append(result, step=3, agent_id="marine_0", metadata={"reward": 1.5})
    output.append(result, step=3, agent_id="marine_1")
    return output


def test_same_execution_and_no_duplicate_evaluation(monkeypatch):
    graph = policy()
    runtime = GraphRuntime(RuntimeConfig(0, 1))
    normal = runtime.traverse(graph, ())
    calls = []
    original = ProgramExecutor.execute

    def execute(self, program, observation):
        calls.append(program.id)
        return original(self, program, observation)

    monkeypatch.setattr(ProgramExecutor, "execute", execute)
    detailed = runtime.traverse_detailed(graph, ())
    assert calls == [10, 19, 19, 12]  # shared program is evaluated once per occurrence
    assert detailed.action_id == normal.action_id == 4
    assert detailed.visited_team_ids == normal.visited_team_ids == (0, 1)
    assert (
        tuple(
            TraversalStep(
                d.team_id,
                d.winner_learner_id,
                d.winner_bid,
                d.winner.atomic_action_id,
                d.winner.referenced_team_id,
            )
            for d in detailed.team_decisions
        )
        == normal.steps
    )
    first, second = detailed.team_decisions
    assert [e.bid for e in first.evaluations] == [2.0, -1.0]
    excluded = second.evaluations[0]
    assert excluded.bid is None
    assert not excluded.eligible and not excluded.evaluated and not excluded.winner
    assert second.winner_learner_id == 9  # first tied learner, not lowest ID
    assert sum(e.winner for e in second.evaluations) == 1
    calls.clear()
    assert runtime.act(graph, ()) == detailed.action_id
    assert calls == [10, 19, 19, 12]


def test_serialization_and_metadata_snapshot(tmp_path):
    trace = session()
    text = dumps_session(trace)
    assert dumps_session(loads_session(text)) == text
    snapshot = snapshot_to_dict(trace.graph)
    assert snapshot_from_dict(snapshot) == trace.graph
    assert json.dumps(snapshot, sort_keys=True) == json.dumps(
        snapshot_to_dict(TraceSession.from_graph(policy()).graph), sort_keys=True
    )
    path = tmp_path / "trace.json"
    trace.to_json(path)
    assert dumps_session(TraceSession.from_json(path)) == text
    metadata = {"nested": [1, {"reward": 2}]}
    frame = trace.append(trace.frames[0].traversal, metadata=metadata)
    metadata["nested"].append(3)
    assert frame.metadata == {"nested": [1, {"reward": 2}]}
    frame.metadata["nested"].append(4)
    assert frame.metadata == {"nested": [1, {"reward": 2}]}


@pytest.mark.parametrize(
    "metadata", [{"x": float("nan")}, {1: "bad"}, {"x": (1,)}, {"x": object()}]
)
def test_invalid_metadata(metadata):
    trace = session()
    with pytest.raises(ValueError):
        trace.append(trace.frames[0].traversal, metadata=metadata)


@pytest.mark.parametrize("mutation", ["version", "winner", "bid", "path", "index"])
def test_corrupt_trace_rejected(mutation):
    data = json.loads(dumps_session(session()))
    if mutation == "version":
        data["schema_version"] = 2
    elif mutation == "index":
        data["frames"][0]["frame_index"] = 1
    else:
        trace = data["frames"][0]["traversal"]
        if mutation == "path":
            trace["action_id"] = 999
        else:
            evaluation = trace["team_decisions"][0]["evaluations"][0]
            evaluation[mutation] = False if mutation == "winner" else None
    with pytest.raises(ValueError):
        loads_session(json.dumps(data))


def test_wrong_topology_rejected():
    trace = session()
    with pytest.raises(ValueError, match="root"):
        trace.append(replace(trace.frames[0].traversal, root_team_id=100))


def test_aggregate_counts_shared_program():
    counts = aggregate(session())
    assert counts.team_visits == {0: 2, 1: 2}
    assert counts.program_evaluations == {10: 2, 19: 4, 12: 2}
    assert counts.program_wins == {10: 2, 19: 2}
    assert counts.learner_evaluations[(0, 9)] == 2
    assert counts.learner_wins[(0, 9)] == 0
    assert counts.learner_wins[(1, 9)] == 2
    assert counts.edge_traversals == {(0, 0): 2, (1, 9): 2}
    assert counts.atomic_actions == {4: 2}


def test_base_import_without_optional_dependencies():
    code = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'plotly', 'networkx'}:
            raise ImportError('blocked optional dependency')
sys.meta_path.insert(0, Block())
import pytpg
import pytpg.runtime
import pytpg.visualization
assert 'plotly' not in sys.modules and 'networkx' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("missing", ["plotly", "networkx"])
def test_optional_dependency_error(monkeypatch, tmp_path, missing):
    import pytpg.visualization.render as rendering

    original = rendering.importlib.import_module

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == missing:
            raise ImportError("blocked")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(rendering.importlib, "import_module", blocked)
    with pytest.raises(VisualizationDependencyError, match=r"pytpg\[visualization\]"):
        render_html(session(), tmp_path / "trace.html")


def test_standalone_html_and_fixed_replay(tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    pytest.importorskip("networkx")
    trace = session()
    figure = build_figure(trace)
    assert len(figure.frames) == 2
    assert figure.frames[0].name != figure.frames[1].name  # repeated environment steps
    for frame in figure.frames:
        assert tuple(frame.data[-1].x) == tuple(figure.data[-1].x)
        assert tuple(frame.data[-1].y) == tuple(figure.data[-1].y)
    assert tuple(build_figure(trace).data[-1].x) == tuple(figure.data[-1].x)
    path = render_html(trace, tmp_path / "trace.html")
    html = path.read_text(encoding="utf-8")
    assert len(html) > 100_000 and "<html>" in html and "Plotly" in html
    import re

    assert not re.search(r'<script[^>]+src=["\']https?://', html)
    monkeypatch.setattr("webbrowser.open", lambda *args: pytest.fail("browser opened"))
    trace.to_json(tmp_path / "trace.json")
    assert (
        main([str(tmp_path / "trace.json"), "--output", str(tmp_path / "cli.html")])
        == 0
    )
    assert (tmp_path / "cli.html").stat().st_size > 100_000


def test_empty_session(tmp_path):
    trace = TraceSession.from_graph(policy())
    assert not loads_session(dumps_session(trace)).frames
    assert not aggregate(trace).team_visits
    pytest.importorskip("plotly")
    assert render_html(trace, tmp_path / "empty.html").exists()


def test_cli_help_does_not_open_browser(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda *args: pytest.fail("browser opened"))
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0


def test_detailed_root_selection_and_limits():
    from pytpg.runtime import RootSelectionError, TraversalLimitExceededError

    graph = policy()
    multiple = replace(graph, root_team_ids=(TeamID(0), TeamID(1)))
    runtime = GraphRuntime(RuntimeConfig(0, 1))
    with pytest.raises(RootSelectionError):
        runtime.traverse_detailed(multiple, ())
    for root in (0, 1):
        normal = runtime.traverse(multiple, (), root_team_id=root)
        detailed = runtime.traverse_detailed(multiple, (), root_team_id=root)
        assert detailed.root_team_id == root
        assert detailed.action_id == normal.action_id
        assert detailed.visited_team_ids == normal.visited_team_ids
    with pytest.raises(TraversalLimitExceededError):
        GraphRuntime(RuntimeConfig(0, 1), max_steps=1).traverse_detailed(graph, ())


def test_replay_clears_previous_path_and_preserves_positions():
    from pytpg.core import InputIndex, InputOperand

    pytest.importorskip("plotly")
    graph = policy()
    root, child = graph.teams
    first, other = root.learners
    dynamic = replace(
        first,
        program=Program(
            first.program.id,
            (
                Instruction(
                    OperatorName("identity"),
                    RegisterIndex(0),
                    (InputOperand(InputIndex(0)),),
                ),
            ),
        ),
    )
    graph = replace(graph, teams=(replace(root, learners=(dynamic, other)), child))
    runtime = GraphRuntime(RuntimeConfig(1, 1))
    trace = TraceSession.from_graph(graph)
    for value in (2.0, -2.0):
        detailed = runtime.traverse_detailed(graph, (value,))
        normal = runtime.traverse(graph, (value,))
        assert detailed.visited_team_ids == normal.visited_team_ids
        trace.append(detailed)
    assert [f.traversal.visited_team_ids for f in trace.frames] == [(0, 1), (0,)]
    figure = build_figure(trace)
    first_frame, second_frame = figure.frames
    labels = tuple(first_frame.data[-1].text)
    child_index = labels.index("T1")
    assert first_frame.data[-1].marker.opacity[child_index] == 1
    assert second_frame.data[-1].marker.opacity[child_index] < 1
    assert first_frame.data[0].line.width == 4
    assert second_frame.data[0].line.width == 1
    assert tuple(first_frame.data[-1].x) == tuple(second_frame.data[-1].x)
    assert tuple(first_frame.data[-1].y) == tuple(second_frame.data[-1].y)


def test_cli_reports_missing_dependencies(monkeypatch, tmp_path, capsys):
    import pytpg.visualization.cli as cli

    path = tmp_path / "trace.json"
    session().to_json(path)

    def unavailable(*args):
        raise VisualizationDependencyError('pip install "pytpg[visualization]"')

    monkeypatch.setattr(cli, "render_html", unavailable)
    with pytest.raises(SystemExit) as error:
        cli.main([str(path)])
    assert error.value.code == 2
    assert 'pip install "pytpg[visualization]"' in capsys.readouterr().err


def test_concentric_layout_and_requested_node_shapes():
    import math

    pytest.importorskip("plotly")
    figure = build_figure(session())
    nodes = figure.data[-1]
    positions = dict(zip(nodes.text, zip(nodes.x, nodes.y, strict=True), strict=True))
    assert positions["T0"] == (0, 0)
    assert math.hypot(*positions["P10 / L0"]) == pytest.approx(1)
    assert math.hypot(*positions["T1"]) == pytest.approx(2)
    assert math.hypot(*positions["A4"]) == pytest.approx(2)
    assert len(set(positions.values())) == len(positions)
    shapes = figure.layout.shapes[-len(nodes.text) :]
    by_label = dict(zip(nodes.text, shapes, strict=True))
    assert by_label["T0"].fillcolor == "#2878A5"
    assert by_label["T0"].type == "path"
    assert "Q" in by_label["T0"].path  # rounded corners
    assert by_label["P10 / L0"].fillcolor == "#78B7D0"
    assert by_label["P10 / L0"].x1 == by_label["P10 / L0"].y1
    assert by_label["A4"].fillcolor == "#8C3A5B"
    assert by_label["A4"].x1 > by_label["A4"].y1  # true ellipse
    for frame in figure.frames:
        assert [s.fillcolor for s in frame.layout.shapes] == [
            s.fillcolor for s in figure.layout.shapes
        ]


def test_radial_layout_handles_unreachable_cycles():
    from pytpg.visualization.layout import radial_layout

    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph([(0, 1), (1, 0), (2, 3), (3, 4), (4, 2)])
    positions, depth = radial_layout(graph, 0)
    assert set(positions) == set(graph)
    assert len(set(positions.values())) == len(graph)
    assert depth == 3
    assert radial_layout(graph, 0) == (positions, depth)
