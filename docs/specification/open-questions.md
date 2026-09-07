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

Learner sharing remains representable. Milestone 3 mutations use clone-on-write
for the selected team membership.

## Resolved in Milestone 3: evolution

Milestone 3 fixes a complete graph as the population unit; finite scalar fitness
maximization; stable tournament ties; stable elitism; asexual mutation-only
reproduction; relative mutation-category weights; explicit no-op attempts;
clone-on-write learner membership; graph-local ID allocation; root-preserving
team deletion with incoming-reference removal and no recursive orphan cleanup;
and one caller-owned NumPy RNG stream. See [Evolution semantics](evolution.md)
and [Mutation semantics](mutation.md).

Persistent genealogy identity, crossover, population-wide sharing,
multi-objective comparison, evaluator RNG ownership, and alternative mutation
probability interpretations remain deferred rather than silently implied.

## Resolved in Milestone 4: reproducible research infrastructure

Milestone 4 fixes immutable composed experiment configuration and canonical
digests; call-order-independent named seed derivation; structured synchronous
lifecycle events; failure-propagating logging callbacks; deterministic fitness
and mutation statistics; recorded environment metadata; normalized version-1
graph JSON; evaluated-boundary version-1 checkpoint JSON; PCG64 state capture;
and exact deterministic continuation without re-evaluating the checkpoint
generation. See [Reproducible research infrastructure](research-infrastructure.md).

Portable custom-operator registries, arbitrary stochastic evaluator state,
callback-private state, full persistent genealogy, and schema migrations remain
open rather than being encoded with pickle or implicit conventions.

## Resolved in Milestone 5: Gymnasium integration

Milestone 5 fixes lazy optional Gymnasium loading, scalar Discrete actions,
row-major flattening of fixed-shape numeric observations, separate terminated
and truncated signals, fresh environments per graph evaluation, common episode
seeds, and an independent hard rollout limit. See
[Gymnasium integration](gymnasium-integration.md).

Composite spaces, vector environments, asynchronous rollouts, and environment
state serialization remain open.

## Resolved in Milestone 6: stateful TPG

Milestone 6 fixes explicit episode reset, memory appended after the raw
observation, one atomic memory update after successful graph traversal,
unchanged zero-initialized learner registers, per-graph evaluation memory, and
oldest-to-newest observation-history ordering. See
[Stateful memory](stateful-memory.md).

Persistent per-learner registers, explicit memory-addressing instructions,
evolved memory-write actions, memory mutation, and mid-episode environment plus
memory checkpoints remain alternative research designs rather than implicit
parts of the reference runtime.

## Resolved in Milestone 7: multi-agent TPG

Milestone 7 fixes an ordered fixed agent roster, exact synchronous observation
sets, isolated per-agent memory for independent control, graph parameter sharing
without live-state sharing, transactional independent steps, centralized joint
observation concatenation, one shared memory update, and reversible row-major
mixed-radix joint actions. See [Multi-agent TPG](multi-agent.md).

Dynamic and turn-based rosters, partial observations, individual-agent episode
termination, environment integration, continuous/composite actions,
centralized-training/decentralized-execution, coevolution, and multi-agent credit
assignment remain explicit future policies.

## Project decisions before public release

- Final distribution name (the current `pytpg` name is provisional).
- Open-source license.
- Citation metadata and archival release process.
- Supported Python version window and compatibility policy.
