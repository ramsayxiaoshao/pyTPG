# Reproducible research infrastructure

This document defines the Milestone 4 reference contracts for experiment
configuration, seed management, lifecycle events, statistics, metadata,
serialization, and checkpoint continuation.

## Experiment configuration

`TPGConfig` is a frozen composition of the existing `InitializationConfig`,
`MutationConfig`, tournament size, and `ReproductionConfig`. It does not replace
those focused values or mix dynamic population state into configuration.

The configuration digest is SHA-256 over canonical JSON with sorted keys,
compact separators, and non-finite numbers forbidden. Readers require the exact
version-1 field set, including fields whose Python constructors have defaults.
This prevents an omitted field from silently acquiring a different default in a
future package release.

## Seed ownership

`SeedManager(master_seed)` derives 128-bit seeds by hashing the master seed,
namespace, and optional non-negative index under the fixed
`pytpg-seed-v1` domain. Named streams are independent of request order. Asking
for an `evolution` stream therefore produces the same sequence whether or not
an `evaluation` stream was requested first.

The low-level `EvolutionEngine` retains the Milestone 3 contract: the caller
supplies its RNG. `run_experiment` uses separate `initialization` and
`evolution` streams. A stochastic custom evaluator SHOULD use a separately
named `evaluation` stream and must define how its private state is checkpointed;
the reference sequential evaluator and toy objectives are deterministic.

## Lifecycle events and logging

The engine emits this ordered lifecycle:

```text
run_started
generation_evaluated
(generation_reproduced, generation_evaluated) repeated N times
run_finished
```

Events contain only structured, environment-independent scalar fields.
Callbacks run synchronously in caller order. Callback exceptions propagate so
a failed research log is never silently ignored. `LoggingCallback` writes one
canonical compact JSON object through a caller-supplied standard-library logger
and never configures global logging or prints from the engine.

## Statistics and metadata

`GenerationStatistics` records population size, finite minimum, maximum, mean,
median, population standard deviation, stable best position, elite count, and
per-operator attempted/applied mutation counts. Statistics are computed from
immutable run results without consuming randomness.

`ExperimentMetadata` records name, content-derived run ID, UTC timestamp, master
seed, configuration digest, pyTPG/Python/NumPy versions, platform, and sorted
key-value tags. Metadata does not participate in evolution. Supplying the same
explicit timestamp and fields produces the same run ID; an automatically
captured timestamp intentionally distinguishes separately launched runs.

## Graph JSON format

Standalone graphs use `format = "pytpg-graph"` and `format_version = 1`.
Programs, learners, and teams are normalized into ordered tables. Teams store
learner IDs, learners store program IDs and tagged actions, and instructions
store tagged operands. This preserves shared learner/program values on load and
supports graph cycles without recursive JSON. Unknown or missing fields,
duplicate table IDs, unused table entries, invalid actions/operands, and
unresolved memberships are rejected.

The model format version is independent of the package version. Unsupported
versions fail explicitly; readers do not guess a migration.

## Checkpoint JSON and continuation

Checkpoints use `format = "pytpg-checkpoint"` and `format_version = 1`. They
contain:

```text
complete TPGConfig
ExperimentMetadata
one evaluated population
reference PCG64 state
accumulated RunStatistics
```

The boundary is deliberately an **evaluated generation**. Resume begins with
reproduction from its stored fitness and does not evaluate that generation a
second time. This makes an uninterrupted deterministic run equivalent to a run
split by save/load, including final population, next RNG state, and accumulated
statistics.

Checkpoint loading verifies the configuration digest, population size,
statistics boundary, PCG64 state, and every graph against the configured
reference runtime and default operator registry. Custom operator implementations
must be supplied through a future versioned registry mechanism before they can
participate in portable checkpoints.

Files are UTF-8 JSON and are written to a temporary sibling followed by atomic
replacement. Pickle is not used. Full reproduction genealogy, callback-private
state, arbitrary evaluator state, and schema migrations remain deferred.
