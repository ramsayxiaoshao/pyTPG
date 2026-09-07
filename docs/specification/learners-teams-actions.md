# Learners, bidding, teams, and actions

## Learner

A learner is the immutable composition of a stable `LearnerID`, a `Program`, and
an `Action`. During execution, its program evaluates the current observation and
register state to produce a scalar bid. Learners do not own random generators and
do not know about environment APIs.

Whether one learner ID may belong to multiple teams as shared graph structure is
unresolved. The representation permits sharing but requires the same ID to denote
the same immutable learner value throughout a graph.

## Action

`AtomicAction(ActionID)` terminates graph traversal. The core treats the ID as an
opaque non-negative integer. An adapter is responsible for mapping it to an
environment-specific action.

`TeamReference(TeamID)` requests traversal to another team. `GraphRuntime`
resolves it through the containing `TPGGraph`; a learner never stores a direct
Python team object.

## Bidding and team selection

A team is a non-empty ordered tuple of learners. The deterministic runtime MUST
execute every learner program independently from zero registers and retain each
raw finite register-0 output as its bid. It selects the greatest bid. When two or
more learners have equal greatest bids, the learner appearing first in the team
tuple wins. Learner IDs do not participate in tie-breaking and selection consumes
no randomness.

`DeterministicRuntime.bids()` preserves team order and
`DeterministicRuntime.select()` returns the winning learner. Repeated evaluation
of equal values and equal observations MUST return equal bids and the same
winner.

`DeterministicRuntime.act()` returns the winning `AtomicAction.action_id`. If the
winner contains a `TeamReference`, the Milestone 1 runtime raises
`TeamReferenceRequiresGraphError`; it MUST NOT guess a traversal policy. The
graph runtime defines reference eligibility and cycle handling in
[Graph traversal and invariants](graph.md). Learner sharing and memory ownership
are independent: Milestone 6 memory belongs to a stateful controller episode,
not to a learner or team. See [Stateful memory](stateful-memory.md).

`Team.act(observation)` is the low-level convenience form. It infers the smallest
register count capable of addressing the team's existing instructions and takes
the observation length as the input size. Reproducible experiment code SHOULD
instead construct and record an explicit `RuntimeConfig` and pass its runtime to
`Team.act`.
