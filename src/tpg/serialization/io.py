"""UTF-8 JSON parsing and recoverable atomic file replacement."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tpg.serialization.errors import InvalidSerializedDataError


def dump_json(value: object) -> str:
    """Return deterministic, human-readable JSON ending with one newline."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def load_json(text: str) -> Any:
    """Parse JSON and translate decoder failures to the package error type."""

    if not isinstance(text, str):
        raise InvalidSerializedDataError("serialized JSON must be text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidSerializedDataError(f"invalid JSON: {error.msg}") from error


def write_atomic(path: str | Path, text: str) -> None:
    """Replace one file atomically after fully writing its temporary sibling."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def read_text(path: str | Path) -> str:
    """Read one UTF-8 serialized artifact."""

    return Path(path).read_text(encoding="utf-8")


__all__ = ["dump_json", "load_json", "read_text", "write_atomic"]
