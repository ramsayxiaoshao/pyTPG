# Evolution semantics

This document defines the Milestone 3 reference evolutionary system. It is a
deliberately small, deterministic baseline, not a claim that these choices are
the only valid TPG algorithm.

## Individual and population

One individual is one complete immutable `TPGGraph`. Team, learner, and program
IDs are local to that graph, so two population members may contain the same IDs
without sharing evolutionary identity. Population-wide module sharing is not
part of this milestone.

A `Population` is an ordered, non-empty tuple of graphs plus a non-negative
generation number. Initialization creates each graph independently from the
same explicit RNG stream. A one-team graph contains atomic learners. A
multi-team graph is initialized as a forward chain rooted at team 0: the first
learner in each non-terminal team refers to the next team, and all remaining
learners are atomic. Consequently every initial team is reachable and retains
an atomic fallback.

Graph-local initialization assigns team, learner, and program IDs from zero in
stable construction order. Mutation allocates above the greatest currently
present graph-local ID. These IDs identify values inside a graph; they are not
permanent genealogy IDs and may be reused after deletion.

## Evaluation and fitness

Fitness is one finite real scalar and is maximized. `NaN`, infinity, booleans,
and non-real values are errors. `SequentialEvaluator` evaluates each graph
exactly once in population order and preserves that order in its result.

The reference engine does not inject randomness into a fitness function. A
reproducible experiment therefore MUST use a deterministic fitness function or
give a stateful evaluator/function its own explicitly managed, recorded random
stream. Parallel evaluation remains an interchangeable future implementation
of the `Evaluator` protocol.

## Selection and reproduction

Tournament selection samples distinct population positions within each
tournament. Different tournaments are independent and may select the same
parent. The greatest fitness wins; equal fitness favors the earlier source
population position without consuming another random value.

Reproduction keeps the best `elite_count` graphs first, in stable ranked order.
Each remaining child has one tournament-selected parent and is produced by
applying exactly `mutation_steps` mutation attempts in sequence. A recorded
no-op still counts as one attempt. There is no crossover in the reference
system. Population size is constant and the generation increases by one.

`ChildRecord` records the new position, source-parent position, elite status,
and ordered mutation outcomes. This is local run instrumentation rather than a
persistent genealogy format.

## Run boundary and reproducibility

`EvolutionEngine.run(..., generations=N)` evaluates the supplied population,
performs exactly `N` reproduction steps, and evaluates the resulting generation
after every step. It therefore returns `N + 1` evaluated populations and `N`
reproduction records. `N = 0` is valid.

All initialization, selection, reproduction, and mutation randomness comes
from the caller-supplied `numpy.random.Generator`. The reference implementation
does not use module-global random state and does not split the stream
implicitly. Given the same package and NumPy versions, configuration, initial
inputs, deterministic fitness function, seed, and call order, the run result is
reproducible.

Milestone 4's higher-level `run_experiment` composes this contract with
call-order-independent named streams: initialization and evolution receive
separate generators derived from one recorded master seed. The low-level engine
contract is unchanged.

## Deliberately deferred alternatives

Milestone 3 does not define crossover, multi-objective comparison,
population-wide shared teams or learners, probabilistic evaluation, fitness
aggregation across episodes, parallel evaluation, persistent genealogy, or
adaptive mutation probabilities. Versioned reference checkpoints are specified
separately in [Reproducible research infrastructure](research-infrastructure.md).
