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
src/pytpg/
    core/                 # Immutable domain model
    runtime/              # Program, team, and graph reference execution
    evolution/            # Population, selection, mutation, reproduction, engine
    evaluation/           # Evaluator protocols and deterministic toy objectives
    callbacks/            # Structured lifecycle events and logging consumers
    serialization/        # Versioned graph and checkpoint JSON
    config.py             # Immutable experiment-level configuration
    seed.py               # Named deterministic random streams
    metadata.py           # Recorded experiment provenance
    experiment.py         # Reproducible run/resume composition
    memory/               # Episode-scoped state and graph-runtime composition
    multiagent/           # Independent and centralized controller composition
    adapters/             # Optional environment boundaries and fitness adapters
docs/
    specification/        # Normative algorithm semantics and open questions
tests/                    # Fast unit, property, regression, integration tests
benchmarks/               # Profiling workloads, not correctness tests
examples/                 # Small runnable examples
```

Milestone 7 publishes `core`, the deterministic `runtime`, immutable
evolutionary state and operators, sequential evaluation, research configuration,
callbacks, statistics, metadata, versioned JSON artifacts, and an optional
Gymnasium adapter. Stateful behavior is added through reusable memory components
without creating stateful copies of core values or graph execution. Multi-agent
behavior composes those same graphs and stateful runtimes without duplicating
the core, traversal, or evolution engines.

## Environment integration

`GymnasiumAdapter` validates the modern reset/step API, flattens fixed-shape
numeric observations, maps zero-based atomic action IDs to scalar Discrete
actions, and owns the environment lifecycle. `GymnasiumFitness` composes that
adapter with the existing `FitnessFunction` protocol; no Gymnasium branch was
added to the evolution engine.

The Gymnasium import is lazy and available only through the `gymnasium` package
extra. Mock-environment tests exercise the adapter without the extra, while a
small CartPole integration test runs when it is installed.

## Stateful memory composition

The `memory` layer depends only on public core and runtime contracts. A
`StatefulGraphRuntime` owns an episode-scoped `Memory`, appends its snapshot to
the raw observation, delegates unchanged traversal to `GraphRuntime`, and then
updates memory once after a successful atomic action. Runtime and core never
import memory.

`NullMemory` preserves stateless behavior, `RegisterMemory` supplies validated
fixed-width state with pluggable updates, and `ObservationHistoryMemory` composes
register memory into a deterministic history. Gymnasium fitness accepts a
memory factory and resets the resulting component before every episode. This
keeps state out of immutable genomes and prevents population members from
sharing live episode state.

## Multi-agent composition

The `multiagent` layer depends only on core, runtime, and memory. An
`IndependentMultiAgentRuntime` owns an ordered tuple of agent controllers and
supports heterogeneous observation widths, scalar action counts, graphs, and
memory. Multiple controllers may share an immutable graph, but never a live
runtime or memory owner. Synchronous independent steps snapshot all memories and
roll the complete group back if any controller fails.

`SharedControlRuntime` concatenates observations in declared roster order and
traverses one graph once. A mixed-radix codec reversibly maps the graph's scalar
atomic action to one scalar action per agent, including heterogeneous action
counts. The layer contains no environment, reward, fitness, or mutation policy;
adapters and experiments compose those decisions at their existing boundaries.

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

Focused frozen configurations define genome shape, initialization, mutation,
and reproduction. `TPGConfig` composes them for one reference experiment and
provides a canonical digest. Dynamic state remains in immutable population,
run-result, statistics, metadata, and checkpoint values rather than the core
graph model.

## Research infrastructure

`SeedManager` derives named RNG streams independently of call order.
`EvolutionEvent` and `EvolutionCallback` isolate lifecycle reporting from the
engine; the standard logging consumer does not configure global logging.
`ExperimentMetadata` records environment provenance, while pure statistics
summarize immutable evolution results.

`run_experiment` and `resume_experiment` are composition helpers rather than a
giant trainer: they build existing initializer, evaluator, reproducer, and
engine components. Serialization uses normalized graph tables and an evaluated
checkpoint boundary. The serializer depends on core/evolution/research values;
core and runtime never depend on serialization.

## Runtime composition

`RegisterFile` provides finite, bounds-checked ephemeral state.
`OperatorRegistry` owns named fixed-arity operators without global mutation.
`ProgramExecutor` validates and executes a single program.
`DeterministicRuntime` independently evaluates every learner and selects a team
winner. `GraphValidator` applies graph-wide invariants, while `GraphRuntime`
performs bounded, visited-aware traversal and produces traces. These layers know
nothing about environments or evolutionary state.

The runtime consumes immutable core values through read-only structural
`Protocol` contracts and does not import `pytpg.core`. `Team.act` is a convenience
method that imports the concrete runtime only when called. This preserves the
declared core-to-runtime dependency while keeping module loading free of a
cycle; an architecture test guards the boundary.

## Execution replay

`runtime.trace` contains immutable detailed execution records. `GraphRuntime`
captures optional records from its existing bidding pass; it does not depend on
the visualization layer. `visualization.model`, `serialization` and `aggregate`
provide dependency-free topology snapshots, versioned trace sessions and usage
counts. `visualization.render` lazily loads NetworkX and Plotly for fixed-layout
offline HTML replay. The CLI is a thin rendering client. Environment callers
supply step labels, agent IDs and JSON metadata without adapter coupling.
