"""Benchmark-specific DSL surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from src.dsl.fields import get_field
from src.dsl.operators import OPERATORS
from src.dsl.validator import validate_expr


@dataclass(frozen=True)
class BenchmarkSpec:
    """Fields and examples required by one factor-discovery benchmark."""

    name: str
    description: str
    target: str
    fields: tuple[str, ...]
    examples: tuple[str, ...]


_BENCHMARK_SPECS = [
    BenchmarkSpec(
        name="daily_cross_sectional_rankic",
        description="Rank assets each day by factor score and compare ranks to next-period returns.",
        target="Cross-sectional RankIC / ICIR over daily WRDS or crypto assets.",
        fields=(
            "crsp.dlyret",
            "crsp.dlyclose",
            "crsp.dlyvol",
            "crsp.dlycap",
            "crypto.returns",
            "crypto.close",
            "crypto.volume",
        ),
        examples=(
            "div(ts_mean(crsp.dlyret(20)), ts_std(crsp.dlyret(20)))",
            "neg(div(ts_mean(crypto.returns(7)), ts_std(crypto.returns(30))))",
            "corr(crsp.dlyret(20), crsp.dlyvol(20))",
        ),
    ),
    BenchmarkSpec(
        name="single_asset_direction",
        description="Score one asset through time and test whether the score predicts its next return sign.",
        target="Single-asset directional accuracy or time-series IC.",
        fields=(
            "crsp.dlyopen",
            "crsp.dlyhigh",
            "crsp.dlylow",
            "crsp.dlyclose",
            "crsp.dlyvol",
            "crsp.dlyret",
            "crypto.open",
            "crypto.high",
            "crypto.low",
            "crypto.close",
            "crypto.volume",
            "crypto.returns",
        ),
        examples=(
            "sign(diff(crypto.close(5)))",
            "sub(last(crsp.dlyclose(1)), ts_mean(crsp.dlyclose(20)))",
            "tanh(div(ts_mean(crypto.returns(12)), ts_std(crypto.returns(48))))",
        ),
    ),
    BenchmarkSpec(
        name="intraday_microstructure_direction",
        description="Use TAQ-derived intraday features to predict short-horizon price direction.",
        target="Intraday directional prediction over TAQ bars.",
        fields=(
            "taq.spread",
            "taq.midret",
            "taq.imbalance",
            "taq.trade_size",
            "taq.trade_count",
            "taq.volume",
        ),
        examples=(
            "neg(ts_mean(taq.spread(30)))",
            "tanh(div(ts_mean(taq.imbalance(20)), ts_std(taq.midret(20))))",
            "corr(taq.midret(30), taq.volume(30))",
        ),
    ),
]

_BENCHMARKS = {spec.name: spec for spec in _BENCHMARK_SPECS}

for _spec in _BENCHMARK_SPECS:
    for _field in _spec.fields:
        get_field(_field)
    for _example in _spec.examples:
        validate_expr(_example)

BENCHMARKS = MappingProxyType(_BENCHMARKS)


def list_benchmarks() -> tuple[str, ...]:
    """Return benchmark DSL surface names."""

    return tuple(sorted(BENCHMARKS))


def get_benchmark(name: str) -> BenchmarkSpec:
    """Return benchmark DSL surface metadata."""

    key = str(name).strip().lower()
    try:
        return BENCHMARKS[key]
    except KeyError as exc:
        valid = ", ".join(list_benchmarks())
        raise ValueError(f"unknown benchmark '{name}'. valid benchmarks: {valid}") from exc


def list_benchmark_examples(name: str) -> tuple[str, ...]:
    """Return validated DSL examples for one benchmark."""

    return get_benchmark(name).examples
