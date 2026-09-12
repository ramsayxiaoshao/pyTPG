"""Check the built public distribution without pytest's source-path injection."""

import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_installed_wheel(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(wheels), str(root)],
        check=True, capture_output=True, text=True,
    )
    wheel, = wheels.glob("*.whl")
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "pytpg/__init__.py" in names
        assert "pytpg/py.typed" in names
        assert all(
            name.startswith(("pytpg/", "pytpg-0.5.0.dist-info/"))
            for name in names
        )
        assert not any(
            part in {"tests", "__pycache__", "build", "dist", "tpg"}
            or part.endswith((".pyc", ".pyo"))
            for name in names for part in name.split("/")
        )
        metadata = archive.read("pytpg-0.5.0.dist-info/METADATA").decode()
        assert "License-Expression: MIT" in metadata
        for filename in ("LICENSE",):
            assert f"License-File: {filename}" in metadata
            assert f"pytpg-0.5.0.dist-info/licenses/{filename}" in names

    installed = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
         "--target", str(installed), str(wheel)],
        check=True, capture_output=True, text=True,
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-I", "-c", """
import importlib
import importlib.metadata
import pathlib
import pkgutil
import sys
sys.path.insert(0, sys.argv[1])
import pytpg
from pytpg.core import TPGGraph
assert pathlib.Path(pytpg.__file__).resolve().is_relative_to(sys.argv[1])
assert pytpg.__version__ == importlib.metadata.version('pytpg') == '0.5.0'
assert TPGGraph.__module__ == 'pytpg.core.graph'
for module in pkgutil.walk_packages(pytpg.__path__, 'pytpg.'):
    importlib.import_module(module.name)
assert not any(name == 'tpg' or name.startswith('tpg.') for name in sys.modules)
""", str(installed)],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
