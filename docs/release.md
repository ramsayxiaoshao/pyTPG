# Preparing pyTPG 0.5.0

GitHub project: https://github.com/ramsayxiaoshao/pyTPG

Distribution and import namespace: `pytpg`. Python requirement: 3.10 or newer.
This preparation does not publish a release or configure publishing credentials.

## Local verification

In a virtual environment, install `python -m pip install -e ".[dev,gymnasium]"`.
Run:

```bash
python -m pytest
python -m ruff check .
python -m pyright
python -m build
python -m twine check dist/*
```

Use an empty `dist` directory for each release. The distribution test builds a
wheel, inspects its contents and MIT metadata, installs it into a temporary
directory, and imports every package module in an isolated Python subprocess
outside the source tree. It requires the development dependencies and pip.

## Compatibility

Replace `import tpg` and `from tpg...` with `import pytpg` and `from pytpg...`.
There is no `tpg` compatibility shim. Update dotted module paths in downstream
configuration, scripts, and plugins. Pickles referencing `tpg` classes may no
longer load; recreate them or migrate them with the previous environment.
The versioned JSON graph/checkpoint format identifiers and RNG seed domain
remain unchanged. The distribution name and version remain `pytpg` / `0.5.0`.

## License

MIT licensed. Copyright (c) 2026 pyTPG contributors.
The full license is included in both the wheel and source distribution.
Package metadata uses the MIT SPDX expression and explicitly includes LICENSE.
