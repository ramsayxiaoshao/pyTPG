# Contributing

pyTPG is specification-first research software. A change to algorithm behavior
must update the relevant document under `docs/specification/` in the same pull
request. Tests should assert semantics, not only benchmark performance.

Core dependencies must point inward:

```text
application/environment -> adapter -> training/evolution -> core -> runtime
```

The core and runtime must not import environment integrations. Optional
integrations must not become required package dependencies.

Before submitting a change, run:

```bash
ruff check .
pyright
pytest
python -m build
python -m twine check dist/*
```

Install development dependencies with `python -m pip install -e ".[dev]"`.
The distribution test also requires pip and builds an installed-wheel smoke test.
See [release preparation](docs/release.md) for compatibility and licensing.

Milestone 7 intentionally uses a fixed synchronous agent roster. Independent
controllers isolate live episode state; centralized shared control concatenates
observations and uses a mixed-radix scalar joint action. Dynamic or turn-based
rosters, per-agent early termination, PettingZoo integration, partial
observations, continuous/composite actions, coevolution, credit assignment,
centralized-training/decentralized-execution, parallel evaluation, persistent
genealogy, custom-operator schema registration, and serialization migrations
remain outside this milestone.
