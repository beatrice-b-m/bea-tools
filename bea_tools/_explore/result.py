"""Standard-library result models for feature exploration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "0.2"


@dataclass(frozen=True)
class ExplorerResult(Mapping[str, Any]):
    """Immutable top-level result with a strict-JSON-compatible payload."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    stability: str = "unstable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stability": self.stability,
            "kind": self.kind,
            **self.payload,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class KeySpec:
    """An explicit, named determinant; required for composite keys."""

    name: str
    columns: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("KeySpec.name must be a non-empty string")
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("KeySpec.columns must not be empty")
        object.__setattr__(self, "columns", columns)
