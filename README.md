# pyTPG

## What is TPG?

pyTPG is a modular research framework for Tangled Program Graphs
(TPG). The project is being developed specification-first so that algorithmic
choices are explicit, testable, and reproducible.

The repository currently contains **Milestone 7**: deterministic runtime and
evolution, reproducible research infrastructure, optional Gymnasium integration,
episode-scoped stateful TPG, and deterministic multi-agent composition.

## Installation

Requires Python 3.10 or newer. After the first public release, install with:

```bash
pip install pytpg
```

For development from a checkout:

```bash
python -m pip install -e ".[dev]"
```

Install the optional Gymnasium integration with:

```bash
python -m pip install "pytpg[gymnasium]"
```

The PyPI distribution and Python import namespace are both `pytpg`.
The GitHub project name is pyTPG.

Code using the previous `tpg` namespace must update imports to `pytpg`;
no compatibility alias is provided. See [release notes](docs/release.md).

## Minimal example

The following manually constructs and executes a one-team TPG graph:

```python
from pytpg.core import (
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
- Explicit memory reset, snapshot, restore, and atomic update contracts.
- Null, fixed-register, and oldest-to-newest observation-history memory.
- Stateful traversal by observation augmentation without duplicated graph logic.
- Independent and heterogeneous controllers with transactional joint steps.
- Safe immutable-graph parameter sharing with per-agent episode state.
- Centralized shared control with mixed-radix heterogeneous joint actions.

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

Run the three-step delayed-signal memory example with:

```bash
python examples/stateful_delayed_signal.py
```

Run heterogeneous independent and centralized shared control with:

```bash
python examples/multiagent_control.py
```

## Development status

Current version: `0.5.0` (Milestone 7, multi-agent TPG, pre-alpha).

Run the local checks with:

```bash
ruff check .
pyright
pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development boundaries.

## License

MIT licensed. Copyright (c) 2026 pyTPG contributors. See [LICENSE](LICENSE).
