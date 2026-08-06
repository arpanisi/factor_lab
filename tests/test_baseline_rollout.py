from unittest.mock import patch

from examples.baseline_rollout import run_crypto_baseline_rollout
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL


def test_run_crypto_baseline_rollout_with_mocked_llms(tmp_path):
    raw_candidates = (
        "div(ts_sum(crypto.returns(30)), ts_std(crypto.returns(30)))",
        "div(ts_mean(crypto.volume(7)), ts_mean(crypto.volume(30)))",
    )

    def fake_generate(task, namespace, count, **kwargs):
        return ("ts_mean(crypto.returns(5))", "future_return(crypto.returns(5))")

    with patch("examples.baseline_rollout.generate_oracle_seed_candidates", return_value=raw_candidates):
        with patch("src.rft.discovery_loop.generate_miner_candidates", side_effect=fake_generate):
            result = run_crypto_baseline_rollout(
                oracle_model="oracle/model",
                miner_model="miner/model",
                panel_path=DEFAULT_CRYPTO_PANEL,
                tickers=("BTC-USD", "ETH-USD", "XRP-USD"),
                oracle_count=2,
                miner_count=2,
                top_k_seeds=1,
                output_dir=tmp_path,
            )

    payload = result["payload"]
    assert payload["oracle_model"] == "oracle/model"
    assert payload["miner_model"] == "miner/model"
    assert payload["seed_count"] == 1
    assert payload["rollout_count"] == 2
    assert result["json_path"].exists()
    assert result["csv_path"].exists()
    assert result["db_path"].exists()
