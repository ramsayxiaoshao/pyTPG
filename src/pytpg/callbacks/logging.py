"""Standard-library logging callback for evolution lifecycle events."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pytpg.callbacks.base import EvolutionEvent


@dataclass(frozen=True, slots=True)
class LoggingCallback:
    """Emit one compact structured JSON message through a supplied logger."""

    logger: logging.Logger
    level: int = logging.INFO

    def __post_init__(self) -> None:
        if not isinstance(self.logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger")
        if isinstance(self.level, bool) or not isinstance(self.level, int):
            raise TypeError("logging level must be an integer")

    def on_event(self, event: EvolutionEvent) -> None:
        """Log without modifying global logging configuration."""

        self.logger.log(
            self.level,
            json.dumps(
                event.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )


@dataclass(slots=True)
class EventRecorder:
    """In-memory callback useful for tests, notebooks, and custom exporters."""

    events: list[EvolutionEvent]

    def __init__(self) -> None:
        self.events = []

    def on_event(self, event: EvolutionEvent) -> None:
        self.events.append(event)


__all__ = ["EventRecorder", "LoggingCallback"]
