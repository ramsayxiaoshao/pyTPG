# Execution visualization

Install the optional renderer:

```bash
pip install "pytpg[visualization]"
```

Recording, JSON persistence and usage analysis work with the base installation.
Neither `import pytpg` nor `import pytpg.visualization` loads Plotly or NetworkX.
Only rendering requires the extra. No server or frontend build is needed.

## Record and replay

Given an existing `graph` and observations:

```python
from pytpg.runtime import GraphRuntime, RuntimeConfig
from pytpg.visualization import TraceSession, aggregate, render_html

runtime = GraphRuntime(RuntimeConfig(input_size=4, register_count=8))
session = TraceSession.from_graph(graph)
for step, observation in enumerate(observations):
    result = runtime.traverse_detailed(graph, observation)
    # Use result.action_id as the policy action; do not call act() again.
    session.append(result, step=step, agent_id="marine_0",
                   metadata={"reward": 0.0, "environment": "SC2"})

session.to_json("episode_trace.json")
restored = TraceSession.from_json("episode_trace.json")
print(aggregate(restored).program_evaluations)
render_html(restored, "trace.html")
```

Set dimensions for your graph, and pass `root_team_id` to `traverse_detailed`
when the graph has multiple roots. The returned frozen `DetailedTraversalResult`
contains `root_team_id`, `action_id`, ordered `team_decisions`, and the derived
`visited_team_ids`. Each `TeamDecisionTrace` contains all learner evaluations
in original Team order and exposes `winner`, `winner_learner_id`,
`winner_program_id`, and `winner_bid`. Its winner includes the selected action
kind and destination. Excluded references have `eligible=False`,
`evaluated=False`, `winner=False`, and `bid=None`.

An **evaluated program** produced a bid in the actual policy execution. A
**winning program** is the first eligible learner with the maximum bid in its
Team. Other evaluated programs lost that decision. The **traversal path** is
the sequence of visited Teams and winning destination edges, ending in an
atomic action. Tracing records the original execution, without executing
programs again or recording register state.

`GraphRuntime.traverse()` still returns the original lightweight
`TraversalResult`; `act()` still returns an integer action. Both use the same
internal traversal as detailed tracing.

## CLI and example

```bash
pytpg-visualize episode_trace.json --output trace.html
python -m pytpg.visualization episode_trace.json --output trace.html --open
python examples/visualize_trace.py
```

`--open` is opt-in. The example creates a small cyclic graph with changing
paths and writes `example_trace.json` and `example_trace.html` in the current
working directory. Generated HTML embeds Plotly JavaScript and works offline.

Play/Pause and the decision slider replay one full policy decision per frame.
The layout uses a deterministic breadth-first hierarchy of the NetworkX graph.
Root Team 0 is centered when declared; otherwise the first declared root is
centered. Programs occupy radius r, their next Teams/actions radius 2r, and
subsequent levels extend outward. Branch sectors keep descendants together;
shared destinations use their shortest depth and cycles remain visible as back
edges. Unreachable components occupy additional sectors. Crowded rings increase
canvas spacing. Positions remain fixed throughout replay; `layout_seed` remains
accepted for API compatibility but is unused by this nonrandom layout.
`build_figure(session)` exposes the Plotly figure for further customization.
Teams are deep blue (#2878A5) rounded rectangles, programs light blue (#78B7D0)
circles, and actions burgundy (#8C3A5B) ellipses. Base colors remain constant:
opacity and thick borders indicate activity; gold borders and edges identify
winners and selected paths. Excluded learners remain dim and are identified
explicitly in hover information.
Hover information gives IDs, eligibility, evaluation, bid, winner, destination,
and whole-session counts. Eligibility is unknown for unvisited Teams.

## JSON and analysis

A session document has `schema_version: 1`, a versioned `topology` object, and
ordered `frames`. The topology stores integer Team, learner, program and action
IDs, root flags, and ordered learner membership. It contains no executable
program instructions or Python object references. Learner nodes are keyed by
parent Team and learner ID because a learner can be shared by several Teams.

Each frame contains a contiguous `frame_index`, an independent `step` label,
optional `agent_id`, metadata object, and detailed traversal. Repeated step
labels are allowed, for example for agents acting at the same environment step.
`dumps_session` / `loads_session` work with JSON strings; `to_json` / `from_json`
work with paths. Serialization is deterministic and rejects unsupported schema
versions, malformed traces, invalid destinations and inconsistent winners.

Metadata accepts JSON objects with string keys, arrays, strings, finite numbers,
booleans and null. It is copied when appended, so later caller mutations cannot
change recorded history. Observations are not recorded automatically; include a
small observation or summary explicitly in metadata if useful. Titles display
at most 160 characters of metadata. An SC2 or VMAS caller can use `agent_id` and
metadata such as reward, episode ID or environment step; there is no environment
integration in this version.

`aggregate(session)` exposes `team_visits`, `program_evaluations`, `program_wins`,
`learner_evaluations`, `learner_wins`, `edge_traversals` and `atomic_actions`.
Program counts accumulate actual evaluations across all learner occurrences.
Learner and edge counters use `(team_id, learner_id)` keys. An edge count records
the selected learner's destination transition (also used for its incoming Team
edge in the viewer). Counts not present in a counter read as zero.

## Version 1 scope

A session describes one fixed topology; create another session after graph
mutation. Trace/topology validation cannot detect changed program instructions
that retain the same IDs. Frames and Plotly data are held in memory; large
sessions produce large HTML files. Dense graphs can produce a large canvas; zoom or scroll to inspect individual branches. There is
no live capture adapter, instruction/register animation, streaming, evolutionary
comparison, or multi-agent comparison dashboard. Existing stateful/multi-agent
wrappers do not yet expose detailed tracing; this API is on `GraphRuntime`.
