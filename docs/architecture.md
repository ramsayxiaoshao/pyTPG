# Architecture

## Dependency rule

The architectural dependency direction is:

```text
Application / Environment
        |
        v
Adapter
        |
        v
Training / Evolution
        |
        v
TPG Core
        |
        v
Runtime
```

An arrow means "may depend on." Core code must never import an adapter or an
environment package. Runtime code defines execution mechanisms and operator
semantics without knowing how observations or actions are produced externally.

## Initial repository structure

```text
src/tpg/
    core/                 # Immutable domain model
    runtime/              # Program, team, and graph reference execution
    evolution/            # Population, selection, mutation, reproduction, engine
    evaluation/           # Evaluator protocols and deterministic toy objectives
    memory/               # Reserved for memory composition
    adapters/             # Reserved for optional environments
    serialization/        # Reserved for versioned model formats
    callbacks/            # Reserved for lifecycle events
docs/
    specification/        # Normative algorithm semantics and open questions
tests/                    # Fast unit, property, regression, integration tests
benchmarks/               # Profiling workloads, not correctness tests
examples/                 # Small runnable examples
```

Milestone 3 publishes `core`, the deterministic `runtime`, immutable evolutionary
state and operators, a sequential evaluator, and deterministic toy objectives.
The other packages remain boundary markers, not speculative interfaces.

## Core object model

Core entities are frozen dataclasses. Mutation will later produce replacement
values through explicit mutation operators. This gives tests stable before/after
values, prevents accidental cross-individual mutation, and keeps randomness out
of domain objects.

Identifiers are `typing.NewType` integer IDs. They are compact and serialize
cleanly while allowing static analysis to distinguish a `TeamID` from a
`ProgramID`. Evolution allocates IDs from graph contents without a global
counter; core constructors never read global state or randomness.

`Action` is a tagged sum type:

```text
AtomicAction(ActionID) | TeamReference(TeamID)
```

Team references use identifiers instead of direct Python references. This
permits cycles without recursive object construction and gives serialization a
stable reference format.

A learner directly contains its immutable `Program` and `Action`; a team directly
contains learners. Sharing the same immutable learner value among teams remains
representable. `TPGGraph` owns an ordered tuple of teams and an ordered tuple of
root IDs. A future serialization layer may normalize repeated learners and
programs into separate tables without changing the domain API.

## Evolution composition

`GenomeFactory` and initialization services construct valid graph individuals
using an explicit NumPy generator. `MutationOperator`, `SelectionStrategy`, and
`Evaluator` are protocols. `Reproducer` composes selection, elitism, and focused
mutation; `EvolutionEngine` only orchestrates bounded evaluate/reproduce steps.
No layer owns hidden random state, imports an environment, or mutates core
values in place.

`Population`, `EvaluatedPopulation`, `ReproductionResult`, and
`EvolutionRunResult` separate dynamic run state from stable configuration.
Fitness evaluation is abstracted from execution scheduling; Milestone 3 ships
only the stable sequential reference evaluator.

## Configuration and dynamic state

Focused frozen configurations currently define genome shape, initialization,
mutation, and reproduction. A consolidated experiment-level `TPGConfig` and
checkpointable `TPGState` remain Milestone 4 work. Neither belongs in the core
graph model.

## Runtime composition

`RegisterFile` provides finite, bounds-checked ephemeral state.
`OperatorRegistry` owns named fixed-arity operators without global mutation.
`ProgramExecutor` validates and executes a single program.
`DeterministicRuntime` independently evaluates every learner and selects a team
winner. `GraphValidator` applies graph-wide invariants, while `GraphRuntime`
performs bounded, visited-aware traversal and produces traces. These layers know
nothing about environments or evolutionary state.

The runtime consumes immutable core values through read-only structural
`Protocol` contracts and does not import `tpg.core`. `Team.act` is a convenience
method that imports the concrete runtime only when called. This preserves the
declared core-to-runtime dependency while keeping module loading free of a
cycle; an architecture test guards the boundary.
