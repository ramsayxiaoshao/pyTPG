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
    evolution/            # Reserved for Milestone 3
    evaluation/           # Reserved for evaluator abstractions
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

Milestone 1 publishes `core` value objects and the `runtime` reference executor.
The other packages remain boundary markers, not speculative interfaces.

## Core object model

Core entities are frozen dataclasses. Mutation will later produce replacement
values through explicit mutation operators. This gives tests stable before/after
values, prevents accidental cross-individual mutation, and keeps randomness out
of domain objects.

Identifiers are `typing.NewType` integer IDs. They are compact and serialize
cleanly while allowing static analysis to distinguish a `TeamID` from a
`ProgramID`. ID allocation belongs to future construction/evolution services;
core constructors never read global state or randomness.

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

## Configuration and dynamic state

Future `TPGConfig` values will contain stable experimental choices. Future
`TPGState` values will contain generation, population, fitness, and history.
Neither belongs in the core graph model, and neither is introduced before its
semantics are known.

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
