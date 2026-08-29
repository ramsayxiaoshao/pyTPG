"""Structured, optional evolution lifecycle callbacks."""

from tpg.callbacks.base import (
    EvolutionCallback,
    EvolutionEvent,
    EvolutionEventKind,
)
from tpg.callbacks.logging import EventRecorder, LoggingCallback

__all__ = [
    "EventRecorder",
    "EvolutionCallback",
    "EvolutionEvent",
    "EvolutionEventKind",
    "LoggingCallback",
]
