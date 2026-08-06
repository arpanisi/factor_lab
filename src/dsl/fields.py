"""Allowed dataset fields for the Factor Lab DSL."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from src.dsl.namespaces import get_namespace


@dataclass(frozen=True)
class FieldSpec:
    """Metadata for a namespaced DSL data field."""

    namespace: str
    name: str
    kind: str
    description: str
    source_column: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.namespace}.{self.name}"


_FIELD_SPECS = [
    FieldSpec("crsp", "dlyopen", "price", "CRSP daily open price.", "dlyopen"),
    FieldSpec("crsp", "dlyhigh", "price", "CRSP daily high price.", "dlyhigh"),
    FieldSpec("crsp", "dlylow", "price", "CRSP daily low price.", "dlylow"),
    FieldSpec("crsp", "dlyclose", "price", "CRSP daily close price.", "dlyclose"),
    FieldSpec("crsp", "dlyvol", "volume", "CRSP daily trading volume.", "dlyvol"),
    FieldSpec("crsp", "dlyret", "return", "CRSP daily total return.", "dlyret"),
    FieldSpec("crsp", "dlycap", "size", "CRSP daily market capitalization.", "dlycap"),
    FieldSpec("taq", "spread", "microstructure", "Quoted or effective spread.", "spread"),
    FieldSpec("taq", "midret", "return", "Midquote return over an intraday bar.", "midret"),
    FieldSpec("taq", "imbalance", "microstructure", "Order-flow or volume imbalance.", "imbalance"),
    FieldSpec("taq", "trade_size", "microstructure", "Average trade size.", "trade_size"),
    FieldSpec("taq", "trade_count", "microstructure", "Number of trades.", "trade_count"),
    FieldSpec("taq", "volume", "volume", "Intraday traded volume.", "volume"),
    FieldSpec("crypto", "open", "price", "Crypto open price.", "open"),
    FieldSpec("crypto", "high", "price", "Crypto high price.", "high"),
    FieldSpec("crypto", "low", "price", "Crypto low price.", "low"),
    FieldSpec("crypto", "close", "price", "Crypto close price.", "close"),
    FieldSpec("crypto", "volume", "volume", "Crypto traded volume.", "volume"),
    FieldSpec("crypto", "returns", "return", "Crypto simple returns.", "returns"),
]

_FIELDS = {spec.full_name: spec for spec in _FIELD_SPECS}

for _spec in _FIELD_SPECS:
    get_namespace(_spec.namespace)

FIELDS = MappingProxyType(_FIELDS)


def list_fields(namespace: str | None = None) -> tuple[str, ...]:
    """Return registered field names, optionally restricted to a namespace."""

    if namespace is None:
        return tuple(sorted(FIELDS))

    ns = get_namespace(namespace).name
    return tuple(sorted(name for name in FIELDS if name.startswith(f"{ns}.")))


def get_field(full_name: str) -> FieldSpec:
    """Return field metadata from a full name like ``crsp.dlyret``."""

    key = str(full_name).strip().lower()
    try:
        return FIELDS[key]
    except KeyError as exc:
        valid_preview = ", ".join(list_fields()[:8])
        raise ValueError(f"unknown field '{full_name}'. examples: {valid_preview}") from exc

