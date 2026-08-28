"""Import-boundary regression tests for the package layers."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "tpg"


def imported_modules(package: Path) -> set[str]:
    modules: set[str] = set()
    for source_path in package.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.add(node.module)
    return modules


def test_runtime_does_not_import_core_or_higher_layers() -> None:
    forbidden = (
        "tpg.core",
        "tpg.evolution",
        "tpg.evaluation",
        "tpg.memory",
        "tpg.adapters",
        "tpg.serialization",
        "tpg.callbacks",
    )

    imports = imported_modules(PACKAGE_ROOT / "runtime")

    assert not {
        module
        for module in imports
        if any(module == item or module.startswith(f"{item}.") for item in forbidden)
    }


def test_core_does_not_import_higher_layers() -> None:
    forbidden = (
        "tpg.evolution",
        "tpg.evaluation",
        "tpg.memory",
        "tpg.adapters",
        "tpg.serialization",
        "tpg.callbacks",
    )

    imports = imported_modules(PACKAGE_ROOT / "core")

    assert not {
        module
        for module in imports
        if any(module == item or module.startswith(f"{item}.") for item in forbidden)
    }
