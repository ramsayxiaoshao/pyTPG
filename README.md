# pyTPG

## What is TPG?

pyTPG is a modular research framework for Tangled Program Graphs
(TPG). The project is being developed specification-first so that algorithmic
choices are explicit, testable, and reproducible.

The repository currently contains **Milestone 1**: semantic specifications,
typed core representations, and a deterministic reference runtime for programs
and individual teams. Graph traversal and evolution are not implemented yet.

## Installation

For development, use Python 3.10 or newer:

```bash
python -m pip install -e ".[dev]"
```

The distribution name is provisional until the first public release.

## Minimal example

The following manually constructs a team whose learner bids on one observation:

```python
from tpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
    InputIndex,
    InputOperand,
    Instruction,
    Learner,
    LearnerID,
    OperatorName,
    Program,
    ProgramID,
    RegisterIndex,
    Team,
    TeamID,
    TPGGraph,
)

instruction = Instruction(
    operator=OperatorName("multiply"),
    destination=RegisterIndex(0),
    operands=(InputOperand(InputIndex(0)), ConstantOperand(2.0)),
)
program = Program(ProgramID(0), (instruction,))
learner = Learner(LearnerID(0), program, AtomicAction(ActionID(0)))
team = Team(TeamID(0), (learner,))
graph = TPGGraph((team,), (team.id,))

action = team.act((0.75,))
assert action == ActionID(0)
```

For recorded experiments, pass an explicit `RuntimeConfig` rather than relying
on the shape inference provided by `Team.act`.

## Main features

- Explicit core concepts rather than trainer-owned dictionaries.
- Immutable value objects and typed identifiers.
- Environment-independent core architecture.
- Extensible operator registry with protected deterministic arithmetic.
- Strict observation, register, instruction, and operator validation.
- Stable raw-bid selection with deterministic tie-breaking.
- Draft specifications for graphs, mutation, and reproducibility.

## Documentation

- [Architecture](docs/architecture.md)
- [Specification index](docs/specification/README.md)
- [Open algorithmic questions](docs/specification/open-questions.md)

## Examples

Run the manual deterministic-team example with:

```bash
python examples/manual_team.py
```

Environment integrations remain intentionally deferred until Milestone 5.

## Development status

Current version: `0.1.0` (Milestone 1, pre-alpha).

Run the local checks with:

```bash
ruff check .
pyright
pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development boundaries.


