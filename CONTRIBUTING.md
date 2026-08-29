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

Milestone 3 intentionally excludes serialization, checkpoints, consolidated
experiment configuration, environment adapters, memory, parallel evaluation,
and experiment logging.
