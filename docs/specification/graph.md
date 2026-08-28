# Graph traversal and invariants

## Graph representation

`TPGGraph` contains an ordered tuple of teams and an ordered tuple of root team
IDs. A traversal starts from one explicitly selected root; root selection itself
belongs to the calling policy or agent API.

Traversal repeatedly evaluates the current team. An `AtomicAction` terminates
with its action ID. A `TeamReference` changes the current team and continues.

## Structural invariants

A structurally valid graph MUST satisfy all of the following:

- at least one team and one root exist;
- team IDs and root IDs are unique;
- every root ID resolves;
- every team reference resolves;
- every team and program is non-empty;
- repeated learner IDs and program IDs denote equal immutable values;
- every instruction uses a known operator with valid arity;
- input and register indices are within configured bounds.

The last two rules require validation against runtime configuration.
`ProgramExecutor.validate()` enforces them for one program in Milestone 1.
`graph.validate(...)` will provide collected graph-wide diagnostics in Milestone
2; graph constructors currently enforce only the context-free subset.

## Cycles and termination

Cycles are representable and are not rejected merely for being cycles. A runtime
MUST nevertheless terminate deterministically. Candidate policies include a
visited-team exclusion rule, a traversal step limit with a specified fallback,
or validation that every eligible path terminates. The policy is unresolved and
must be fixed before graph traversal is implemented.

The following also remain unresolved:

- whether all teams must be reachable from some root;
- whether every root must have a possible path to an atomic action;
- whether orphaned teams are invalid, warnings, or population bookkeeping;
- the formal definition of a root team during and after mutation; and
- whether self-references are allowed.

Until those decisions are made, no reachability or cycle property is claimed as
a validity invariant.
