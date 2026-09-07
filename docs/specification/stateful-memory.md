# Stateful memory specification

Status: **Milestone 6 implemented**.

This document defines the reference stateful TPG semantics. Memory is composed
around an unchanged graph runtime; it is not stored in `TPGGraph`, `Team`,
`Learner`, `Program`, or `Instruction` values.

## State ownership and episode lifecycle

A `Memory` instance belongs to exactly one controller episode at a time. A
stateful controller MUST be reset before its first decision and MUST be reset
between episodes. Reset restores the memory component's configured initial
state and revision zero. Ending an episode makes another decision illegal until
the next reset.

Population graphs MUST NOT share live memory state during fitness evaluation.
An evaluator MAY reuse one memory object across sequential episodes only when it
resets that object before every episode.

Generation-boundary checkpoints contain graphs and evolutionary state, not a
running environment or mid-episode memory. Milestone 6 therefore makes no claim
that an environment episode can be resumed from a research checkpoint.

## Decision semantics

For a raw environment observation `o_t` and memory snapshot `m_t`, the stateful
controller MUST construct:

```text
augmented_observation_t = o_t + m_t
```

where `+` is tuple concatenation. Raw observation values precede memory values.
Both sequences MUST have their configured exact lengths and contain only finite
real numbers. The composed graph runtime's `input_size` MUST equal:

```text
raw_observation_size + memory_size
```

The unchanged `GraphRuntime` traverses the graph once using that augmented
observation. Only after traversal returns an atomic action does the controller
call the memory update exactly once with the raw observation, atomic action, and
complete traversal result. A failed traversal MUST NOT invoke memory update.

The returned stateful result exposes the augmented observation, traversal, and
memory snapshots immediately before and after the update.

## Program registers are unchanged

Milestone 1's program registers remain zero-initialized and private to each
learner execution. Memory values are input operands on the next decision; they
are not implicit initial register values. This preserves stateless graph
semantics and avoids a second copy of program, team, or traversal execution.

Persistent per-learner registers and new instructions that explicitly address
memory are plausible alternative TPG variants. They are not silently treated as
equivalent to the reference observation-augmentation model.

## Memory protocol

A memory component exposes a fixed non-negative `size` and these operations:

- `reset()` restores the episode initial state;
- `snapshot()` returns immutable finite values and a non-negative revision;
- `update(context)` atomically derives the next state after a successful
  decision; and
- `restore(snapshot)` restores a compatible previously captured state.

Custom memory implementations MUST keep `size` stable for their lifetime. An
update either completes with one valid new snapshot or leaves the prior state
unchanged.

## Reference components

`NullMemory` has size zero and never changes. Composing it with
`StatefulGraphRuntime` is behaviorally equivalent to stateless traversal after
the required episode reset.

`RegisterMemory` owns a fixed tuple of finite values. A supplied `MemoryUpdater`
receives the previous snapshot and decision context and returns the entire next
value tuple. Candidate values are fully validated before they replace the
current state. With no updater, transition updates retain the values but still
advance the revision; callers may use explicit atomic `write()` operations.

`ObservationHistoryMemory` composes `RegisterMemory`. With raw observation width
`W` and history depth `D`, it exposes `W * D` values ordered from oldest to most
recent. Reset fills all positions with zero. After each successful decision it
drops the oldest width-`W` frame and appends the current raw observation. Thus a
decision never sees the observation from its own subsequent update.

## Gymnasium evaluation

`GymnasiumFitness` accepts an optional `MemoryFactory`. It creates one memory
component per graph evaluation, resets it for every seeded episode, and uses the
same episode seeds as the stateless evaluator. Omitting the factory preserves
the Milestone 5 behavior through `NullMemory`.
