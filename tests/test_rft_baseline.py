import io
import json
from unittest.mock import patch

import pandas as pd

from src.rft import (
    MinerConfig,
    build_miner_messages,
    generate_miner_candidates,
    realize_factor,
    run_discovery_for_task,
)
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return io.BytesIO(json.dumps(self.payload).encode("utf-8"))

    def __exit__(self, *args):
        return False


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
    scenario = FactorScenario.from_benchmark(
        "single_asset_direction",
        market="crypto",
        horizon=1,
    )
    seed = SeedCandidate("sign(diff(crypto.close(2)))", 0.5, "sig")
    return build_task_bank([seed], scenario, [EvaluationWindow("2024-01-01", "2024-01-07")])[0]


def test_build_miner_messages_include_task_payload():
    task = _task()

    messages = build_miner_messages(task, namespace="crypto", count=3)
    user = messages[1]["content"]

    assert "sign(diff(crypto.close(2)))" in user
    assert "Generate 3" in user
    assert "crypto.close" in user


def test_realize_factor_validates_candidate():
    valid = realize_factor("ts_mean(crypto.returns(5))")
    invalid = realize_factor("future_return(crypto.returns(5))")

    assert valid.valid
    assert valid.signature
    assert not invalid.valid
    assert invalid.reason


def test_generate_miner_candidates_calls_openrouter(monkeypatch):
    task = _task()
    payload = {"choices": [{"message": {"content": "<expr>ts_mean(crypto.returns(5))</expr>"}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    with patch("src.llm.openrouter.load_dotenv"):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)) as urlopen:
            out = generate_miner_candidates(
                task,
                namespace="crypto",
                count=1,
                config=MinerConfig(model="test/model", timeout_seconds=1),
            )

    body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
    assert body["model"] == "test/model"
    assert out == ("ts_mean(crypto.returns(5))",)


def test_run_discovery_for_task_scores_generated_candidates():
    task = _task()
    frame = _crypto_frame([10, 11, 12, 11, 10, 12, 13])

    def generator(task, namespace, count):
        return ("sign(diff(crypto.close(2)))", "future_return(crypto.close(2))")

    results = run_discovery_for_task(
        task,
        namespace="crypto",
        data=frame,
        price_col="close",
        count=2,
        min_history=2,
        generator=generator,
    )

    assert len(results) == 2
    assert results[0].valid
    assert "directional_accuracy" in results[0].metrics
    assert not results[1].valid
