import pandas as pd
import pytest

from src.rft import MinedFactorDatabase
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank
from src.training import FactorRewardRuntime, make_grpo_reward_func
from src.verl_integration.dataset import build_verl_prompt_rows
from src.verl_integration.reward_bridge import FactorLabVerlRewardBridge
from src.verl_integration.reward_function import reward_fn
from src.verl_integration.verl_main import (
    build_verl_config,
    build_verl_overrides,
    load_factor_verl_config,
)


def _task():
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="test_verl",
    )
    return build_task_bank(
        [SeedCandidate("ts_mean(crypto.returns(2))", 0.1, "sig")],
        scenario,
        [EvaluationWindow("2024-01-01", "2024-01-10")],
    )[0]


def test_build_verl_prompt_rows_uses_rule_reward_schema():
    rows = build_verl_prompt_rows([_task()], namespace="crypto", repeats=2)

    assert len(rows) == 2
    record = rows[0].to_record()
    assert record["reward_model"]["style"] == "rule"
    assert record["extra_info"]["seed_expr"] == "ts_mean(crypto.returns(2))"
    assert record["extra_info"]["namespace"] == "crypto"
    assert record["extra_info"]["family"] == "crypto_daily_cross_sectional_rankic"
    assert record["extra_info"]["start_date"] == "2024-01-01"
    assert record["extra_info"]["end_date"] == "2024-01-10"
    assert record["extra_info"]["bar_minutes"] == 1440
    assert record["extra_info"]["horizon_bars"] == 1
    assert record["prompt"][0]["role"] == "system"


def test_verl_reward_fn_matches_grpo_reward_runtime(monkeypatch, tmp_path):
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

    frames = {
        f"asset_{i}": frame([10, 10 + i * 0.1 + 1, 10 + i * 0.2 + 2, 10 + i * 0.3 + 3, 10 + i * 0.4 + 4, 10 + i * 0.5 + 5, 10 + i * 0.6 + 6, 10 + i * 0.7 + 7])
        for i in range(8)
    }
    panel = {
        "open": pd.concat({k: v["open"] for k, v in frames.items()}, axis=1),
        "high": pd.concat({k: v["high"] for k, v in frames.items()}, axis=1),
        "low": pd.concat({k: v["low"] for k, v in frames.items()}, axis=1),
        "close": pd.concat({k: v["close"] for k, v in frames.items()}, axis=1),
        "volume": pd.concat({k: v["volume"] for k, v in frames.items()}, axis=1),
        "returns": pd.concat({k: v["returns"] for k, v in frames.items()}, axis=1),
    }
    panel_path = tmp_path / "panel.pkl"
    pd.to_pickle(panel, panel_path)
    monkeypatch.setenv("FACTOR_LAB_CRYPTO_PANEL", str(panel_path))
    monkeypatch.setenv("FACTOR_LAB_TICKERS", ",".join(frames))
    monkeypatch.setenv("FACTOR_LAB_ARCHIVE_JSONL", str(tmp_path / "archive.jsonl"))

    extra_info = build_verl_prompt_rows([_task()], namespace="crypto")[0].extra_info
    rewards = reward_fn(
        ["<expr>ts_mean(crypto.returns(2))</expr>", "bad output"],
        extra_info=extra_info,
    )

    assert len(rewards) == 2
    assert float(rewards[0]) > -1.0
    assert float(rewards[1]) == -1.0


def test_verl_reward_bridge_places_reward_on_last_response_token(monkeypatch):
    torch = pytest.importorskip("torch")

    def fake_reward_fn(completions, **metadata):
        assert metadata["extra_info"]["task_id"] == "demo"
        return torch.tensor([0.25, -1.0], dtype=torch.float32)

    monkeypatch.setattr("src.verl_integration.reward_bridge.reward_fn", fake_reward_fn)

    class Batch:
        batch = {
            "responses": torch.zeros((2, 4), dtype=torch.long),
            "prompts": torch.zeros((2, 3), dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=torch.long,
            ),
        }
        non_tensor_batch = {
            "responses_str": ["<expr>ts_mean(crypto.returns(2))</expr>", "bad output"],
            "extra_info": {"task_id": "demo"},
        }

    out = FactorLabVerlRewardBridge()(Batch(), return_dict=True)

    reward_tensor = out["reward_tensor"]
    assert reward_tensor.shape == (2, 4)
    assert reward_tensor[0, 1] == pytest.approx(0.25)
    assert reward_tensor[1, 3] == pytest.approx(-1.0)
    assert reward_tensor[0, 0] == 0.0
    assert out["reward_extra_info"]["expr"][0] == "ts_mean(crypto.returns(2))"


def test_verl_config_merge_enables_validation():
    raw_config = load_factor_verl_config("config/verl_qwen3_14b_fullft_a100.yaml")
    overrides = build_verl_overrides(raw_config)

    assert overrides["trainer"]["val_before_train"] is True
    assert overrides["trainer"]["test_freq"] == 20
    assert overrides["trainer"]["save_freq"] == 20

    merged = build_verl_config(raw_config, base_config="config/verl_ppo_trainer_base.yaml")
    assert merged.trainer.val_before_train is True
    assert merged.trainer.test_freq == 20
    assert merged.trainer.save_freq == 20
