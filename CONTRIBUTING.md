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

Milestone 1 intentionally excludes team-reference graph traversal, evolution,
serialization, environment adapters, memory, and experiment logging.
