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

## Required before Milestone 2: graph runtime

1. **Cycle handling:** visited-team exclusion, hard step budget, both, or another
   rule; and the fallback if no learner remains eligible.
2. **Termination invariant:** must every root have some atomic path, must every
   possible path terminate under policy, or is bounded fallback sufficient?
3. **Root definition:** explicitly stored roots, teams with zero incoming learner
   references, or a population-level property?
4. **Orphans:** validation error, warning, retained evolutionary material, or
   garbage collected structure?
5. **Self-references:** valid cycle edges or prohibited structures?
6. **Root selection:** one graph per root agent, caller-selected root, or a
   deterministic graph-level default?
7. **Learner sharing:** may the same learner ID belong to multiple teams, and if
   so does mutation clone it or intentionally affect every owner?

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
