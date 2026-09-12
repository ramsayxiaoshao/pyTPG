"""Import-boundary regression tests for the package layers."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "pytpg"


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
        "pytpg.core",
        "pytpg.evolution",
        "pytpg.evaluation",
        "pytpg.memory",
        "pytpg.multiagent",
        "pytpg.adapters",
        "pytpg.serialization",
        "pytpg.callbacks",
    )

    imports = imported_modules(PACKAGE_ROOT / "runtime")

    assert not {
        module
        for module in imports
        if any(module == item or module.startswith(f"{item}.") for item in forbidden)
    }


def test_core_does_not_import_higher_layers() -> None:
    forbidden = (
        "pytpg.evolution",
        "pytpg.evaluation",
        "pytpg.memory",
        "pytpg.multiagent",
        "pytpg.adapters",
        "pytpg.serialization",
        "pytpg.callbacks",
    )

    imports = imported_modules(PACKAGE_ROOT / "core")

    assert not {
        module
        for module in imports
        if any(module == item or module.startswith(f"{item}.") for item in forbidden)
    }


def test_environment_packages_are_confined_to_adapters() -> None:
    protected_layers = (
        "runtime",
        "core",
        "evolution",
        "evaluation",
        "memory",
        "multiagent",
    )

    for layer in protected_layers:
        imports = imported_modules(PACKAGE_ROOT / layer)
        assert not {
            module
            for module in imports
            if module == "gymnasium" or module.startswith("gymnasium.")
        }


def test_memory_depends_only_on_core_and_runtime_layers() -> None:
    forbidden = (
        "pytpg.adapters",
        "pytpg.callbacks",
        "pytpg.evaluation",
        "pytpg.evolution",
        "pytpg.multiagent",
        "pytpg.serialization",
    )

    imports = imported_modules(PACKAGE_ROOT / "memory")

    assert not {
        module
        for module in imports
        if any(module == item or module.startswith(f"{item}.") for item in forbidden)
    }


def test_multiagent_depends_only_on_core_runtime_and_memory() -> None:
    forbidden = (
        "pytpg.adapters",
        "pytpg.callbacks",
        "pytpg.evaluation",
        "pytpg.evolution",
        "pytpg.serialization",
    )

    imports = imported_modules(PACKAGE_ROOT / "multiagent")

    assert not {
        module
        for module in imports
        if any(module == item or module.startswith(f"{item}.") for item in forbidden)
    }
