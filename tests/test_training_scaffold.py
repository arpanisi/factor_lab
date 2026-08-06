import pandas as pd

from src.rft import MinedFactorDatabase
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank
from src.training import FactorRewardRuntime, make_grpo_reward_func


def _task():
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="test_training",
    )
    return build_task_bank(
        [SeedCandidate("ts_mean(crypto.returns(2))", 0.1, "sig")],
        scenario,
        [EvaluationWindow("2024-01-01", "2024-01-10")],
    )[0]


def _frames():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")

    def frame(vals):
        close = pd.Series(vals, index=idx, dtype=float)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000.0,
                "returns": close.pct_change(),
            }
        )

    return {
        "a": frame([10, 11, 12, 13, 14, 15, 16, 17]),
        "b": frame([10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5]),
        "c": frame([10, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8, 8.6]),
    }


def test_grpo_reward_function_returns_one_float_per_completion():
    task = _task()
    runtime = FactorRewardRuntime(
        task=task,
        namespace="crypto",
        data=_frames(),
        price_col="close",
        archive=MinedFactorDatabase(),
        min_history=3,
        min_assets=3,
    )

    reward_func = make_grpo_reward_func(runtime)
    rewards = reward_func(["<expr>ts_mean(crypto.returns(2))</expr>", "bad output"])

    assert len(rewards) == 2
    assert rewards[0] > -1.0
    assert rewards[1] == -1.0


def test_grpo_reward_function_can_log_completion_results(tmp_path):
    task = _task()
    log_path = tmp_path / "reward_log.jsonl"
    runtime = FactorRewardRuntime(
        task=task,
        namespace="crypto",
        data=_frames(),
        price_col="close",
        archive=MinedFactorDatabase(),
        min_history=3,
        min_assets=3,
        reward_log_jsonl=log_path,
    )

    reward_func = make_grpo_reward_func(runtime)
    rewards = reward_func(["<expr>ts_mean(crypto.returns(2))</expr>", "bad output"])

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert str(rewards[0]) in lines[0]
    assert "bad output" in lines[1]
