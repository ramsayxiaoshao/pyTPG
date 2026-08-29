# Core object model

## Identifiers

The model defines distinct integer types for `ActionID`, `InputIndex`,
`LearnerID`, `ProgramID`, `RegisterIndex`, and `TeamID`. Values MUST be
non-negative. IDs identify entities within one graph/model artifact; they are not
process-global IDs. Index values address an ordered runtime collection.

ID allocation MUST be explicit and deterministic for a fixed construction seed.
Allocation is outside the domain value constructors.

## Values and ownership

`Instruction`, operand values, `Program`, actions, `Learner`, `Team`, and
`TPGGraph` are immutable values. Evolution MUST create replacement values rather
than change an existing value in place.

The Python representation is:

```python
Instruction(operator, destination, operands)
Program(id, instructions)
Learner(id, program, action)
AtomicAction(action_id)
TeamReference(team_id)
Action = AtomicAction | TeamReference
Team(id, learners)
TPGGraph(teams, root_team_ids)
```

Tuples preserve order. Team learner order is therefore defined data and resolves
equal-bid ties. Team and program IDs exist to support stable references,
inspection, genealogy, and versioned serialization.

## Locally enforced invariants

- IDs and indices MUST be non-negative.
- Operator names MUST be non-empty.
- Constant operands MUST be finite numbers.
- Programs MUST contain at least one instruction.
- Teams MUST contain at least one learner and MUST NOT repeat a learner ID within
  the same team.
- A graph MUST contain at least one team and at least one root.
- Team IDs and root IDs MUST be unique within a graph.
- Every root and every team reference MUST resolve to a graph team.
- If one learner or program ID appears more than once in a graph, it MUST denote
  an equal immutable value each time.

Operator existence, operator arity, and register/input bounds require runtime
configuration and are not constructor-level invariants. Milestone 1 enforces
them through `ProgramExecutor.validate()` and before every execution.

Graph-wide invariants requiring reachability and action semantics are enforced
by `GraphValidator` in Milestone 2. In particular, every valid team has an atomic
learner and direct self-references are rejected. Orphans produce warnings rather
than constructor failures or automatic deletion.
