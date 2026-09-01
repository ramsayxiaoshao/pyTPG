# pyTPG

## What is TPG?

pyTPG is a modular research framework for Tangled Program Graphs
(TPG). The project is being developed specification-first so that algorithmic
choices are explicit, testable, and reproducible.

The repository currently contains **Milestone 5**: deterministic runtime and
evolution, reproducible research infrastructure, and optional Gymnasium
integration through a strict environment adapter.

## Installation

For development, use Python 3.10 or newer:

```bash
python -m pip install -e ".[dev]"
```

Install the optional Gymnasium integration with:

```bash
python -m pip install -e ".[gymnasium]"
```

The distribution name is provisional until the first public release.

## Minimal example

The following manually constructs and executes a one-team TPG graph:

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

action = graph.act((0.75,))
assert action == ActionID(0)
```

For recorded experiments, pass an explicit `GraphRuntime(RuntimeConfig(...))`
rather than relying on the shape inference provided by `graph.act`.

## Main features

- Explicit core concepts rather than trainer-owned dictionaries.
- Immutable value objects and typed identifiers.
- Environment-independent core architecture.
- Extensible operator registry with protected deterministic arithmetic.
- Strict observation, register, instruction, and operator validation.
- Stable raw-bid selection with deterministic tie-breaking.
- Validated Team-reference traversal with cycle and step-limit safety.
- Graph diagnostics, traversal traces, and structural summaries.
- Explicit NumPy RNG ownership and fixed-seed reproducibility.
- Valid connected population initialization and finite scalar evaluation.
- Stable tournament selection, elitism, and mutation-only reproduction.
- Eight focused clone-on-write mutation operators with invariant validation.
- Inspectable parent and mutation records for each offspring generation.
- Canonical experiment configuration digests and named RNG streams.
- Structured lifecycle callbacks, standard logging, and generation statistics.
- Recorded Python/NumPy/package/platform experiment metadata.
- Versioned JSON graph serialization and evaluated-boundary checkpoints.
- Exact deterministic continuation without re-evaluating saved fitness.
- Lazy optional Gymnasium integration with no environment dependency in core.
- Reproducible multi-episode fitness with common seeds and a hard rollout bound.
- Strict modern `terminated`/`truncated` handling and Discrete action mapping.

## Documentation

- [Architecture](docs/architecture.md)
- [Specification index](docs/specification/README.md)
- [Open algorithmic questions](docs/specification/open-questions.md)

## Examples

Run the manual deterministic-team example with:

```bash
python examples/manual_team.py
```

Run the cyclic graph traversal example with:

```bash
python examples/manual_graph.py
```

Run a fixed-seed toy evolution with:

```bash
python examples/evolve_bandit.py
```

Run, save, load, and resume a reproducible experiment with:

```bash
python examples/reproducible_experiment.py
```

After installing the Gymnasium extra, run the small CartPole evolution with:

```bash
python examples/evolve_cartpole.py
```

## Development status

Current version: `0.3.0` (Milestone 5, optional Gymnasium integration,
pre-alpha).

Run the local checks with:

```bash
ruff check .
pyright
pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development boundaries.
