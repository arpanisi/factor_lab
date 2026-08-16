"""Scenario refinement from raw user text."""

from __future__ import annotations

from dataclasses import dataclass

from src.seeds.scenario import FactorScenario


@dataclass(frozen=True)
class RefinedScenario:
    """Structured scenario plus the namespace and price field to score."""

    scenario: FactorScenario
    namespace: str
    price_col: str


def refine_scenario(raw: str) -> RefinedScenario:
    """Convert a raw factor-mining request into a structured scenario.

    This is intentionally conservative and deterministic. An LLM-backed
    refinement function can be added later, but this gives the pipeline a
    reproducible baseline.
    """

    text = str(raw).strip().lower()
    if not text:
        raise ValueError("raw scenario is empty")

    if any(token in text for token in ("taq", "intraday", "microstructure", "spread", "imbalance")):
        return RefinedScenario(
            scenario=FactorScenario.from_benchmark(
                "intraday_microstructure_direction",
                market="wrds_taq",
                horizon=_extract_horizon(text, default=1),
                constraints=("use only point-in-time TAQ-derived fields",),
            ),
            namespace="taq",
            price_col="close",
        )

    if "crypto" in text or any(token in text for token in ("btc", "eth", "coin")):
        benchmark = "daily_cross_sectional_rankic" if _looks_cross_sectional(text) else "single_asset_direction"
        return RefinedScenario(
            scenario=FactorScenario.from_benchmark(
                benchmark,
                market="crypto",
                horizon=_extract_horizon(text, default=1),
                constraints=("use only point-in-time crypto OHLCV fields",),
            ),
            namespace="crypto",
            price_col="close",
        )

    benchmark = "daily_cross_sectional_rankic" if _looks_cross_sectional(text) else "single_asset_direction"
    return RefinedScenario(
        scenario=FactorScenario.from_benchmark(
            benchmark,
            market="wrds_crsp",
            horizon=_extract_horizon(text, default=1),
            constraints=("use only point-in-time WRDS CRSP daily fields",),
        ),
        namespace="crsp",
        price_col="dlyclose",
    )


def _looks_cross_sectional(text: str) -> bool:
    return any(token in text for token in ("cross-sectional", "cross sectional", "rankic", "rank", "many assets"))


def _extract_horizon(text: str, *, default: int) -> int:
    for token in text.replace("-", " ").split():
        if token.endswith("d") and token[:-1].isdigit():
            return int(token[:-1])
        if token.isdigit():
            value = int(token)
            if 1 <= value <= 252:
                return value
    return int(default)
