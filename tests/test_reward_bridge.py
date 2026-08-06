import pandas as pd

from src.rft import MinedFactorDatabase, reward_completion, reward_completions
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank


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


def _task():
    scenario = FactorScenario.from_benchmark("single_asset_direction", market="crypto", horizon=1)
    seed = SeedCandidate("sign(diff(crypto.close(2)))", 0.5, "seed-sig")
    return build_task_bank([seed], scenario, [EvaluationWindow("2024-01-01", "2024-01-07")])[0]


def test_reward_completion_scores_valid_expression():
    result = reward_completion(
        "<expr>sign(diff(crypto.close(2)))</expr>",
        task=_task(),
        namespace="crypto",
        data=_crypto_frame([10, 11, 12, 11, 10, 12, 13]),
        price_col="close",
        min_history=2,
    )

    assert result.valid
    assert result.expr == "sign(diff(crypto.close(2)))"
    assert result.reward is not None
    assert "directional_accuracy" in result.metrics


def test_reward_completion_rejects_invalid_output():
    result = reward_completion(
        "not an expr",
        task=_task(),
        namespace="crypto",
        data=_crypto_frame([10, 11, 12, 11]),
        price_col="close",
    )

    assert not result.valid
    assert result.reward == -1.0
    assert result.reason


def test_reward_completions_batches_and_stores_archive_records():
    archive = MinedFactorDatabase()
    results = reward_completions(
        [
            "<expr>sign(diff(crypto.close(2)))</expr>",
            "<expr>future_return(crypto.close(2))</expr>",
        ],
        task=_task(),
        namespace="crypto",
        data=_crypto_frame([10, 11, 12, 11, 10, 12, 13]),
        price_col="close",
        archive=archive,
        min_history=2,
    )

    assert len(results) == 2
    assert results[0].valid
    assert results[0].stored
    assert not results[1].valid
    assert len(archive.records) == 1
