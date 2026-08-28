# TPG semantic specification

Status: **Milestone 1 runtime semantics implemented; later milestones draft**.

These documents separate settled structural rules from algorithm choices that
must be resolved before the corresponding implementation milestone. Normative
terms such as **MUST**, **MUST NOT**, and **SHOULD** describe intended contracts.
Text marked **unresolved** is not an implementation contract.

- [Core object model](object-model.md)
- [Instructions, registers, and programs](instructions-and-programs.md)
- [Learners, bidding, teams, and actions](learners-teams-actions.md)
- [Graph traversal and invariants](graph.md)
- [Mutation semantics](mutation.md)
- [Open algorithmic questions](open-questions.md)

Specification changes that alter observable behavior require regression tests
once the affected behavior exists.
