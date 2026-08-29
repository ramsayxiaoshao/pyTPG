# Mutation semantics

Mutation operators are focused immutable transformations with the common
contract:

```python
outcome = operator.mutate(graph, rng)
```

The input graph MUST remain unchanged. `MutationOutcome` contains the resulting
graph, the operator name, whether a change was applied, and a human-readable
description. Every changed candidate is validated against the factory's runtime
shape and operator registry before it can enter a population. An implementation
that produces an invalid graph raises `MutationInvariantError`.

## Selection and no-op policy

`WeightedMutation` selects exactly one component operator per attempt using
finite, non-negative relative weights. At least one weight MUST be positive.
Weights are not independent mutation probabilities and are not normalized in
configuration.

An operator with no legal candidate returns the original graph as an explicit
`applied=False` outcome. It does not retry, substitute another category, or
consume random values merely to disguise the no-op. Composite reproduction
records and counts that attempt normally.

## Clone-on-write and IDs

The graph and all core values are immutable. Instruction changes create a new
program and learner; action changes create a new learner and retain its program.
If an immutable learner appears in more than one team, only the selected team
membership is replaced. This clone-on-write rule prevents an apparently local
mutation from changing every shared membership.

New IDs are allocated above the greatest ID currently present in the graph.
They are deterministic and graph-local; no process-global counter is used.

## Reference operators

- **Instruction insertion** chooses a program below its maximum length, inserts
  one freshly generated instruction at any position, and clones its learner.
- **Instruction deletion** chooses a program above its minimum length, deletes
  one instruction, and clones its learner.
- **Instruction modification** replaces one instruction with a freshly
  generated valid instruction and clones its learner.
- **Learner action mutation** chooses uniformly from all alternative atomic
  actions and non-self team references. The last atomic learner in a team may
  change only to another atomic action.
- **Learner addition** inserts one new atomic learner into a team below its
  maximum size.
- **Learner deletion** removes one learner only if the team remains at or above
  its minimum size and retains an atomic learner.
- **Team addition** creates a new all-atomic team and inserts one new reference
  to it into a reachable team with spare capacity. Thus the new team is
  reachable immediately and the source retains its previous atomic fallback.
- **Team deletion** never deletes a root. It removes every incoming reference to
  the selected non-root team, but only when each changed source remains at or
  above its minimum size. It does not recursively delete teams that become
  orphaned; orphan diagnostics remain warnings under the graph specification.

Candidate memberships and mutation choices use stable graph order followed by
the supplied RNG. A mutation never repairs unrelated pre-existing graph errors;
evolution expects valid input graphs.
