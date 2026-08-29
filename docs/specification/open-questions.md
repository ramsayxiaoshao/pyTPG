# Open algorithmic questions

These decisions materially affect the algorithm and must be resolved before the
listed milestone. The default implementation must not be inferred from benchmark
performance alone.

## Resolved in Milestone 1: deterministic runtime

Milestone 1 fixes zero-initialized per-execution registers; exact-length finite
one-dimensional observations; register 0 as the raw finite bid; stable
first-in-team tie-breaking; strict index bounds; non-empty programs; and the
protected operator policy documented in
[Instructions, registers, and programs](instructions-and-programs.md). These are
now reference-runtime contracts rather than open questions.

## Resolved in Milestone 2: graph runtime

Milestone 2 fixes explicit stored roots; required root selection for multi-root
graphs; visited-team reference exclusion; a hard traversal limit; at least one
atomic learner per valid team; self-references as errors; non-self cycles as
valid; and orphan teams as warnings rather than automatic deletion. Exact rules
and the termination argument are documented in
[Graph traversal and invariants](graph.md).

Learner sharing remains representable, but mutation behavior for shared learners
is still a Milestone 3 decision.

## Required before Milestone 3: evolution

1. Population unit: root team, graph, or another individual boundary.
2. Parent selection and fitness comparison, including ties and multi-objective
   extension points.
3. Mutation probabilities, their conditional/unconditional interpretation, and
   retry/no-op behavior.
4. Team deletion, reference repair, root preservation, and orphan cleanup.
5. Deterministic ID allocation and RNG stream ownership across initialization,
   evaluation, selection, reproduction, and mutation.
6. Genealogy event boundaries and the meaning of parentage for shared structure.

## Project decisions before public release

- Final distribution name (the current `pytpg` name is provisional).
- Open-source license.
- Citation metadata and archival release process.
- Supported Python version window and compatibility policy.
