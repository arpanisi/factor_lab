from src.seeds import (
    EvaluationWindow,
    FactorScenario,
    SeedPoolConfig,
    build_scenario_seed_bank,
    build_seed_pool,
    build_task_bank,
    candidate_templates_for_scenario,
    canonical_signature,
    make_time_windows,
    refine_scenario,
)
import pandas as pd


def _crypto_frame(close):
    close = pd.Series(close, index=pd.date_range("2024-01-01", periods=len(close), freq="D"), dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
            "returns": close.pct_change(),
        }
    )


def test_scenario_from_benchmark_uses_benchmark_surface():
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="wrds_crsp",
        horizon=1,
    )

    assert scenario.benchmark == "daily_cross_sectional_rankic"
    assert scenario.market == "wrds_crsp"
    assert "crsp.dlyret" in scenario.fields
    assert "RankIC" in scenario.objective


def test_refine_scenario_maps_raw_text_to_structured_crypto_rankic():
    refined = refine_scenario("Find crypto cross-sectional RankIC factors with 1d horizon")

    assert refined.scenario.market == "crypto"
    assert refined.scenario.benchmark == "daily_cross_sectional_rankic"
    assert refined.namespace == "crypto"
    assert refined.price_col == "close"


def test_refine_scenario_maps_taq_text_to_intraday_surface():
    refined = refine_scenario("Use TAQ spread and imbalance for intraday direction")

    assert refined.scenario.benchmark == "intraday_microstructure_direction"
    assert refined.namespace == "taq"
    assert refined.price_col == "close"


def test_make_time_windows_splits_range():
    windows = make_time_windows("2020-01-01", "2020-12-31", periods=2)

    assert len(windows) == 2
    assert windows[0].start == "2020-01-01"
    assert windows[-1].end == "2020-12-31"


def test_candidate_templates_validate_for_scenario():
    scenario = FactorScenario.from_benchmark(
        "intraday_microstructure_direction",
        market="wrds_taq",
        horizon=1,
    )

    candidates = candidate_templates_for_scenario(scenario)

    assert "taq.spread" in " ".join(candidates)
    assert len(candidates) >= 3


def test_canonical_signature_deduplicates_commutative_forms():
    left = "add(ts_mean(crsp.dlyret(20)), ts_std(crsp.dlyret(20)))"
    right = "add(ts_std(crsp.dlyret(20)), ts_mean(crsp.dlyret(20)))"

    assert canonical_signature(left) == canonical_signature(right)


def test_build_seed_pool_validates_scores_filters_and_deduplicates():
    raw = [
        "add(ts_mean(crsp.dlyret(20)), ts_std(crsp.dlyret(20)))",
        "add(ts_std(crsp.dlyret(20)), ts_mean(crsp.dlyret(20)))",
        "future_return(crsp.dlyret(20))",
        "ts_mean(crsp.dlyret(5))",
    ]
    scores = {
        raw[0]: 0.8,
        raw[1]: 0.7,
        raw[3]: 0.2,
    }

    seeds = build_seed_pool(
        raw,
        scores=scores,
        config=SeedPoolConfig(top_k=5, quality_threshold=0.5),
    )

    assert len(seeds) == 1
    assert seeds[0].expr == raw[0]
    assert seeds[0].score == 0.8


def test_build_task_bank_is_seed_window_cartesian_product():
    scenario = FactorScenario.from_benchmark(
        "single_asset_direction",
        market="crypto",
        horizon=1,
    )
    seeds = build_seed_pool(
        ["ts_mean(crypto.returns(5))", "ts_std(crypto.returns(20))"],
        scores={"ts_mean(crypto.returns(5))": 0.2, "ts_std(crypto.returns(20))": 0.1},
    )
    windows = [
        EvaluationWindow("2020-01-01", "2020-12-31"),
        EvaluationWindow("2021-01-01", "2021-12-31"),
    ]

    tasks = build_task_bank(seeds, scenario, windows)

    assert len(tasks) == 4
    assert tasks[0].seed_expr == "ts_mean(crypto.returns(5))"
    assert tasks[0].window.start == "2020-01-01"
    assert tasks[-1].window.end == "2021-12-31"


def test_build_scenario_seed_bank_scores_and_filters_by_namespace():
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
    )
    frames = {
        "winner": _crypto_frame([10, 11, 12, 13, 14, 15, 16]),
        "middle": _crypto_frame([10, 10.5, 11, 11.5, 12, 12.5, 13]),
        "loser": _crypto_frame([10, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8]),
    }
    windows = [EvaluationWindow("2024-01-01", "2024-01-07")]

    result = build_scenario_seed_bank(
        scenario,
        namespace="crypto",
        data=frames,
        price_col="close",
        windows=windows,
        min_history=3,
        min_assets=3,
        pool_config=SeedPoolConfig(top_k=2, quality_threshold=-1.0),
    )

    assert result.raw_candidate_count > result.scored_candidate_count
    assert len(result.seeds) <= 2
    assert len(result.tasks) == len(result.seeds)
    assert all("crypto." in seed.expr for seed in result.seeds)


def test_build_scenario_seed_bank_handles_single_asset_direction():
    scenario = FactorScenario.from_benchmark(
        "single_asset_direction",
        market="crypto",
        horizon=1,
    )
    frame = _crypto_frame([10, 11, 12, 11, 10, 12, 13])
    windows = [
        EvaluationWindow("2024-01-01", "2024-01-04"),
        EvaluationWindow("2024-01-05", "2024-01-07"),
    ]

    result = build_scenario_seed_bank(
        scenario,
        namespace="crypto",
        data=frame,
        price_col="close",
        windows=windows,
        min_history=2,
        pool_config=SeedPoolConfig(top_k=3, quality_threshold=0.0),
    )

    assert len(result.seeds) > 0
    assert len(result.tasks) == len(result.seeds) * 2


def test_taq_scenario_scoring_uses_close_price():
    refined = refine_scenario("TAQ microstructure intraday direction")
    idx = pd.date_range("2024-01-01 09:30", periods=10, freq="1min")
    taq_frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "volume": [1000.0] * 10,
            "spread": [0.02] * 10,
            "midret": [0.001] * 10,
            "imbalance": [0.1] * 10,
            "trade_size": [100.0] * 10,
            "trade_count": [10.0] * 10,
        },
        index=idx,
    )
    windows = [EvaluationWindow("2024-01-01", "2024-01-01")]

    result = build_scenario_seed_bank(
        refined.scenario,
        namespace=refined.namespace,
        data=taq_frame,
        price_col=refined.price_col,
        windows=windows,
        raw_candidates=("taq.spread(1)", "taq.imbalance(1)"),
        min_history=2,
        pool_config=SeedPoolConfig(top_k=2, quality_threshold=-1.0),
    )
    assert refined.price_col == "close"
    assert len(result.seeds) == 2
    for seed in result.seeds:
        assert seed.score > -1.0

