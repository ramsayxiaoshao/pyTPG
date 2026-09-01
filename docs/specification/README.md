# TPG semantic specification

Status: **Milestone 5 Gymnasium integration implemented**.

These documents separate settled structural rules from algorithm choices that
must be resolved before the corresponding implementation milestone. Normative
terms such as **MUST**, **MUST NOT**, and **SHOULD** describe intended contracts.
Text marked **unresolved** is not an implementation contract.

- [Core object model](object-model.md)
- [Instructions, registers, and programs](instructions-and-programs.md)
- [Learners, bidding, teams, and actions](learners-teams-actions.md)
- [Graph traversal and invariants](graph.md)
- [Evolution semantics](evolution.md)
- [Mutation semantics](mutation.md)
- [Reproducible research infrastructure](research-infrastructure.md)
- [Gymnasium integration](gymnasium-integration.md)
- [Open algorithmic questions](open-questions.md)

Specification changes that alter observable behavior require regression tests
once the affected behavior exists.
