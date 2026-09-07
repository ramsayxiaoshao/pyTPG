# Multi-agent TPG specification

Status: **Milestone 7 implemented**.

This document defines a deterministic, environment-independent composition for
multiple TPG-controlled agents. It reuses immutable `TPGGraph`, graph traversal,
and episode memory rather than introducing multi-agent variants of teams,
learners, programs, mutation, or evolution.

## Fixed synchronous roster

An `AgentSpec` declares a non-empty string ID, an exact non-negative observation
width, and a positive scalar action count. A controller roster MUST be a
non-empty ordered tuple with unique agent IDs. Tuple order is the canonical
order for observation concatenation, controller execution, action decoding,
traces, and error reporting; caller mapping iteration order has no effect.

Every synchronous step MUST supply exactly one observation for every declared
agent. Each observation MUST have its declared exact width and contain only
finite real numbers. Missing and unexpected agents are errors. Dynamic rosters,
agents terminating at different steps, turn-based execution, and environment
agent-name translation are intentionally not inferred by the reference layer.

## Independent and heterogeneous control

A `TPGAgentController` binds one `AgentSpec`, one graph, one selected root, and
one `StatefulGraphRuntime`. Its graph's complete set of atomic action IDs MUST
fit `[0, action_count)`. A multi-root graph requires an explicit root selection.

`IndependentMultiAgentRuntime` executes bound controllers in roster order.
Different controllers MAY have different observation widths, action counts,
graphs, operator registries, and memory implementations. This is the reference
heterogeneous-control model.

Multiple controllers MAY point to the same immutable `TPGGraph`. This is
parameter sharing, not shared live execution state. Controllers MUST NOT share
a `StatefulGraphRuntime` or `Memory` instance: every agent owns its episode
state independently.

All observations are validated before the first graph traverses. Before a step,
the runtime snapshots every agent memory. If any controller traversal, action
check, or memory update fails, it restores every snapshot and re-raises the
original failure. A successful independent step therefore advances all agent
memories exactly once; a failed step advances none.

## Centralized shared control

`SharedControlRuntime` owns one graph, one selected root, and one episode memory
for the complete roster. On each step it constructs the joint observation by
concatenating raw agent observations in roster order:

```text
joint_observation = observation(agent_0) + ... + observation(agent_n)
```

The existing `StatefulGraphRuntime` appends the one shared memory snapshot and
traverses the graph exactly once. Shared memory is consequently updated once
per joint decision with the complete joint observation.

The graph emits one encoded joint action. For action counts `(r0, ..., rn)`,
`JointActionCodec` uses a row-major mixed-radix encoding:

```text
encoded = (((action_0 * r1) + action_1) * r2 + ...) + action_n
```

The graph action space is therefore `[0, product(action_counts))`. Decoding is
reversible and returns actions in roster order. This representation supports
heterogeneous scalar action counts without embedding environment objects in the
TPG runtime.

## Episode lifecycle and ownership

Independent and shared-control runtimes MUST be reset before their first step
and MUST end or reset between episodes. Bound controllers MUST NOT be operated
directly while owned by a multi-agent runtime; external lifecycle changes are
reported as state errors rather than silently repaired.

Per-agent and centralized memory semantics are exactly those of Milestone 6.
Graph and evolutionary values remain immutable. Independent controllers can be
evolved as separate populations using each agent's `action_count`; a
centralized graph can be evolved by using the codec product as its action count
and the summed observation width plus memory width as its input size. No branch
is required in the existing evolution engine.

## Deferred environment policy

Milestone 7 does not add a PettingZoo or multi-agent Gymnasium dependency. An
adapter must define how environment IDs map to `AgentID`, how rewards become
fitness, whether agents can disappear mid-episode, and how termination and
truncation aggregate. Dynamic/turn-based rosters, partial observations,
continuous or composite actions, centralized-training/decentralized-execution,
coevolution, credit assignment, and multi-agent checkpoint schemas remain
explicit future policies.
