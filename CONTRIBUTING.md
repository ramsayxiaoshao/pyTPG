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
```

Milestone 5 intentionally supports only scalar Discrete actions and fixed-shape
numeric observations. Composite-space transforms, vector environments, memory,
parallel evaluation, arbitrary stochastic-evaluator checkpoints, persistent
genealogy, custom-operator schema registration, and serialization migrations
remain outside this milestone.
