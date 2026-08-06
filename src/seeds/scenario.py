"""Structured factor-mining scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from src.dsl.benchmarks import get_benchmark


@dataclass(frozen=True)
class FactorScenario:
    """A structured version of a factor-mining request."""

    name: str
    benchmark: str
    market: str
    horizon: int
    objective: str
    fields: tuple[str, ...]
    constraints: tuple[str, ...] = ()

    @classmethod
    def from_benchmark(
        cls,
        benchmark: str,
        *,
        market: str,
        horizon: int,
        name: str | None = None,
        constraints: tuple[str, ...] = (),
    ) -> "FactorScenario":
        spec = get_benchmark(benchmark)
        return cls(
            name=name or spec.name,
            benchmark=spec.name,
            market=market,
            horizon=int(horizon),
            objective=spec.target,
            fields=spec.fields,
            constraints=constraints,
        )
