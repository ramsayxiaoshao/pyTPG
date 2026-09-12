"""Public package namespace for the pyTPG research framework."""

from importlib.metadata import version as _version

__version__ = _version("pytpg")

__all__ = ["__version__"]
