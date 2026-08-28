# Mutation semantics

Mutation is deferred to Milestone 3. This document fixes its architectural
boundary without selecting probabilities or repair behavior.

Each mutation operator will implement one focused transformation conceptually
equivalent to:

```python
class MutationOperator(Protocol[T]):
    def mutate(self, individual: T, rng: numpy.random.Generator) -> T: ...
```

Operators MUST receive an explicit random generator, MUST NOT mutate the input
value in place, and MUST either return a value satisfying their documented
postconditions or report failure explicitly. Composite mutation will sequence
named operators rather than combine unrelated behavior in one method.

Expected categories include instruction insertion, deletion, and modification;
program mutation; learner action mutation; team membership mutation; and team
reference mutation. Each category requires unit tests. Once graph mutation
exists, property-based tests SHOULD verify that a structurally valid input graph
remains valid after every successful valid mutation.

Unresolved semantics include minimum/maximum program length, probability
interpretation, retry limits, cloning versus shared-learner updates, reference
repair, orphan cleanup, root preservation, and whether a no-op is a successful
mutation.

