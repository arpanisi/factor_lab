"""Dataset namespace registry for the Factor Lab DSL."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class NamespaceSpec:
    """Metadata for a DSL data namespace."""

    name: str
    description: str
    time_scale: str


_NAMESPACES = {
    "crsp": NamespaceSpec(
        name="crsp",
        description="WRDS CRSP daily equity fields.",
        time_scale="daily",
    ),
    "taq": NamespaceSpec(
        name="taq",
        description="WRDS TAQ intraday microstructure fields or derived bars.",
        time_scale="intraday",
    ),
    "crypto": NamespaceSpec(
        name="crypto",
        description="Crypto OHLCV fields from local exchange/Yahoo-style data.",
        time_scale="configurable",
    ),
}

NAMESPACES = MappingProxyType(_NAMESPACES)


def list_namespaces() -> tuple[str, ...]:
    """Return registered DSL namespace names."""

    return tuple(sorted(NAMESPACES))


def get_namespace(name: str) -> NamespaceSpec:
    """Return namespace metadata or raise for unknown namespaces."""

    key = str(name).strip().lower()
    try:
        return NAMESPACES[key]
    except KeyError as exc:
        valid = ", ".join(list_namespaces())
        raise ValueError(f"unknown namespace '{name}'. valid namespaces: {valid}") from exc

