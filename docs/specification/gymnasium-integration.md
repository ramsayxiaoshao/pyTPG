# Gymnasium integration specification

Status: **Milestone 5 implemented**.

This document defines the optional boundary between Gymnasium environments and
the environment-independent TPG graph runtime. It does not change core graph,
program, bidding, traversal, mutation, or selection semantics.

## Dependency boundary

`pytpg.adapters` MUST remain importable when Gymnasium is absent. Gymnasium MUST
be loaded only when an environment is created, and it MUST be distributed as an
optional extra. `pytpg.core`, `pytpg.runtime`, `pytpg.evolution`, and `pytpg.evaluation`
MUST NOT import Gymnasium.

An environment factory MUST return a fresh environment for each graph fitness
evaluation. The evaluator MUST close that environment on both success and
failure. This prevents episode or wrapper state from leaking between graphs.

## Supported spaces

The reference adapter supports:

- a scalar `Discrete`-like action space with positive `n`;
- the optional non-zero `Discrete.start` offset;
- a fixed-shape integer or floating-point observation space;
- scalar numeric observations, represented to TPG as a length-one tuple; and
- fixed-shape observations, flattened in C row-major order.

TPG atomic action IDs are always zero-based in `[0, n)`. The adapter maps them
to the environment action `Discrete.start + action_id`. A graph containing any
atomic action outside `[0, n)` MUST fail before rollout.

Composite `Dict`, `Tuple`, `Sequence`, and other variable-shape observations are
not silently encoded. They require a future explicit observation transform or
a task-specific adapter. Non-finite observation and reward values are rejected.

## Modern episode API

Reset MUST use and validate the modern result:

```text
(observation, info)
```

Step MUST use and validate the modern result:

```text
(observation, reward, terminated, truncated, info)
```

`terminated` and `truncated` remain separate inspectable values. Either ends an
episode and requires reset before another step. The adapter does not recover the
deprecated single `done` signal.

## Reproducible fitness

`GymnasiumEvaluationConfig` contains the exact ordered episode seeds and an
independent positive hard step limit. `from_seed` derives episode seeds from the
recorded master seed through the `gymnasium_episode` namespace.

Every graph in an evolutionary evaluation receives the same ordered episode
seeds (common random numbers). Fitness is the arithmetic mean of total episode
reward. Episode totals use stable finite summation for aggregation. Each
episode result records:

- seed;
- total reward;
- executed step count;
- environment termination;
- environment truncation; and
- whether the adapter's independent hard limit ended the rollout.

The hard limit is a safety boundary and is reported separately; it MUST NOT be
misreported as Gymnasium `truncated=True` because the environment did not emit
that signal.

Environment implementation details can still introduce nondeterminism. The
contract here controls TPG randomness and reset seeds; reproducibility claims
for a particular task additionally depend on that environment's own contract.

## Stateful extension

`GymnasiumFitness` MAY receive a `MemoryFactory`. One fresh memory component is
created for each graph evaluation and reset before every episode. The graph then
receives the adapter's flattened observation followed by the current memory
values. Omitting the factory composes `NullMemory` and preserves the stateless
Milestone 5 result. Exact memory semantics are specified in
[Stateful memory](stateful-memory.md).
