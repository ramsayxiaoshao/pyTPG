"""Immutable environment and provenance metadata for research runs."""

from __future__ import annotations

import hashlib
import importlib
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tpg import __version__
from tpg.config import TPGConfig
from tpg.evolution.errors import EvolutionConfigurationError


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Recorded provenance that does not participate in evolutionary behavior."""

    name: str
    run_id: str
    created_at_utc: str
    master_seed: int
    config_digest: str
    package_version: str
    python_version: str
    numpy_version: str
    platform: str
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "run_id",
            "created_at_utc",
            "config_digest",
            "package_version",
            "python_version",
            "numpy_version",
            "platform",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise EvolutionConfigurationError(
                    f"metadata {field_name} must be a non-empty string"
                )
        if len(self.run_id) != 24 or any(
            character not in "0123456789abcdef" for character in self.run_id
        ):
            raise EvolutionConfigurationError(
                "metadata run_id must be a 24-character lowercase hex ID"
            )
        if (
            isinstance(self.master_seed, bool)
            or not isinstance(self.master_seed, int)
            or self.master_seed < 0
        ):
            raise EvolutionConfigurationError(
                "metadata master_seed must be non-negative"
            )
        if len(self.config_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_digest
        ):
            raise EvolutionConfigurationError(
                "metadata config_digest must be a lowercase SHA-256 digest"
            )
        try:
            timestamp = datetime.fromisoformat(self.created_at_utc)
        except ValueError as error:
            raise EvolutionConfigurationError(
                "metadata created_at_utc must be an ISO-8601 timestamp"
            ) from error
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise EvolutionConfigurationError(
                "metadata created_at_utc must use the UTC offset"
            )
        if not isinstance(self.tags, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in self.tags
        ):
            raise EvolutionConfigurationError(
                "metadata tags must be non-empty string pairs"
            )
        keys = tuple(key for key, _ in self.tags)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise EvolutionConfigurationError(
                "metadata tags must have unique keys in sorted order"
            )

    @classmethod
    def capture(
        cls,
        name: str,
        master_seed: int,
        config: TPGConfig,
        *,
        tags: tuple[tuple[str, str], ...] = (),
        created_at_utc: str | None = None,
    ) -> ExperimentMetadata:
        """Capture versions and platform with a deterministic content-derived ID."""

        timestamp = (
            datetime.now(timezone.utc).isoformat()
            if created_at_utc is None
            else created_at_utc
        )
        normalized_tags = tuple(sorted(tags))
        identity = (
            f"{name}\0{master_seed}\0{config.digest}\0{timestamp}\0{normalized_tags}"
        ).encode()
        numpy: Any = importlib.import_module("numpy")
        return cls(
            name=name,
            run_id=hashlib.sha256(identity).hexdigest()[:24],
            created_at_utc=timestamp,
            master_seed=master_seed,
            config_digest=config.digest,
            package_version=__version__,
            python_version=platform.python_version(),
            numpy_version=str(numpy.__version__),
            platform=platform.platform(),
            tags=normalized_tags,
        )


__all__ = ["ExperimentMetadata"]
