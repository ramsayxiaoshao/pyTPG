# Graph traversal and invariants

## Graph representation and roots

`TPGGraph` contains an ordered tuple of teams and an ordered tuple of explicit
root team IDs. Roots are model data; they are not inferred from incoming edge
counts. A graph with one root uses it by default. A graph with multiple roots
requires the caller to pass one of its declared root IDs. A non-root team MUST
NOT be selected as a traversal root.

Team references remain ID-based and are resolved through the graph. Non-self
directed cycles are valid TPG structure. Direct self-references are invalid
because the visited-team policy below makes them permanently ineligible.

## Graph validation

`graph.validate()` returns a `GraphValidationReport` containing stable coded
errors and warnings. It does not mutate or repair the graph. `report.is_valid`
means that no error was found; `report.require_valid()` raises
`GraphValidationError` and retains the complete report.

A valid graph MUST satisfy all of the following:

- at least one team and one declared root exist;
- team IDs and root IDs are unique;
- every root and every team reference resolves;
- every team and program is non-empty;
- every team contains at least one `AtomicAction` learner;
- no learner references its own containing team;
- repeated learner IDs and program IDs denote equal immutable values;
- every instruction uses a registered operator with valid arity; and
- input, destination-register, and register-operand indices are within the
  selected `RuntimeConfig`.

Calling `graph.validate()` without a config infers the minimum input and register
dimensions capable of addressing the existing instructions. This is useful for
structural inspection but cannot prove compatibility with an experiment's real
observation shape. Experiment code SHOULD pass its explicit `RuntimeConfig`.
`GraphRuntime` always validates against its exact configuration by default.

A team unreachable from every root is an `orphan_team` warning, not an error.
Orphans remain available as evolutionary material and are never silently
deleted. Their programs and local invariants are still validated.

## Traversal

Traversal starts with an empty visited set and the selected root. At each step,
`GraphRuntime` MUST:

1. add the current team ID to `visited`;
2. consider every atomic learner eligible;
3. consider a team-reference learner eligible only when its target is not in
   `visited`;
4. execute only eligible learner programs using Milestone 1 bidding semantics;
5. select the greatest bid, resolving ties by original team tuple order;
6. return an atomic action immediately, or continue at the selected unvisited
   team; and
7. append the decision to an immutable `TraversalStep` trace.

`TraversalStep` stores atomic action IDs and referenced team IDs in distinct
fields. `TraversalResult` contains the terminal action and the complete ordered
trace. `GraphRuntime.act()` returns only the terminal action ID.

## Termination guarantee

Every valid team contains an atomic learner. A traversal never revisits a team,
so every selected team reference strictly increases the number of visited teams.
At every team, at least its atomic learner remains eligible. Therefore a valid
graph with `N` teams MUST return an atomic action in at most `N` decisions.

The runtime additionally applies a hard step limit. Its default is the number of
teams; callers may select a smaller positive limit. Reaching the limit raises
`TraversalLimitExceededError`. When validation is explicitly disabled, malformed
graphs may instead raise `NoEligibleLearnerError` or `MissingTeamError`, but they
still MUST NOT loop indefinitely.

## Inspection

`graph.summary()` reports root, total, reachable, orphan, and cyclic team counts;
ID-deduplicated learner and program counts; instruction totals and mean program
length; and atomic/reference learner counts. Cycles are reported as structure,
not treated as validation failures.

Graph visualization/export, maximum-depth conventions for cyclic graphs, and
mutation-time orphan cleanup remain later-milestone work.
