"""Initial seed candidate templates."""

from __future__ import annotations

from src.dsl.benchmarks import get_benchmark
from src.seeds.scenario import FactorScenario


_EXTRA_TEMPLATES = {
    "daily_cross_sectional_rankic": (
        "div(ts_sum(crsp.dlyret(60)), ts_std(crsp.dlyret(60)))",
        "neg(ts_mean(crsp.dlyret(5)))",
        "div(ts_mean(crsp.dlyvol(20)), ts_mean(crsp.dlyvol(120)))",
        "neg(log(crsp.dlycap(1)))",
        "div(ts_sum(crypto.returns(30)), ts_std(crypto.returns(30)))",
        "div(ts_mean(crypto.volume(7)), ts_mean(crypto.volume(30)))",
    ),
    "single_asset_direction": (
        "sub(last(crypto.close(1)), ts_mean(crypto.close(20)))",
        "div(sub(last(crypto.close(1)), first(crypto.close(20))), ts_std(crypto.returns(20)))",
        "tanh(ts_sum(crypto.returns(12)))",
        "sign(sub(last(crsp.dlyclose(1)), ts_mean(crsp.dlyclose(10))))",
        "div(ts_mean(crsp.dlyret(5)), ts_std(crsp.dlyret(20)))",
    ),
    "intraday_microstructure_direction": (
        "div(ts_mean(taq.trade_count(30)), ts_mean(taq.trade_count(120)))",
        "div(ts_mean(taq.volume(30)), ts_mean(taq.volume(120)))",
        "neg(div(ts_mean(taq.spread(20)), ts_std(taq.spread(60))))",
        "tanh(ts_sum(taq.midret(10)))",
        "mul(ts_mean(taq.imbalance(20)), neg(ts_mean(taq.spread(20))))",
    ),
}


def candidate_templates_for_scenario(scenario: FactorScenario) -> tuple[str, ...]:
    """Return raw seed candidates for a structured scenario."""

    benchmark = get_benchmark(scenario.benchmark)
    return benchmark.examples + _EXTRA_TEMPLATES.get(benchmark.name, ())
