import io
import json
from pathlib import Path
from unittest.mock import patch

from src.seeds import FactorScenario
from src.seeds.oracle import (
    OpenRouterConfig,
    build_oracle_seed_messages,
    extract_seed_expressions,
    generate_oracle_seed_candidates,
    generate_oracle_seed_candidates_from_file,
)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return io.BytesIO(json.dumps(self.payload).encode("utf-8"))

    def __exit__(self, *args):
        return False


def test_build_oracle_seed_messages_are_namespace_specific():
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
    )

    messages = build_oracle_seed_messages(scenario, namespace="crypto", count=8)
    user = messages[1]["content"]

    assert "Generate 8" in user
    assert "crypto.returns" in user
    assert "crsp.dlyret" not in user.split("Allowed fields: ", 1)[1].split("\n", 1)[0]


def test_extract_seed_expressions_prefers_expr_tags():
    text = "<expr>ts_mean(crypto.returns(5))</expr>\n<expr>ts_std(crypto.returns(20))</expr>"

    assert extract_seed_expressions(text) == (
        "ts_mean(crypto.returns(5))",
        "ts_std(crypto.returns(20))",
    )


def test_generate_oracle_seed_candidates_from_file_filters_invalid_and_wrong_namespace(tmp_path: Path):
    path = tmp_path / "oracle.txt"
    path.write_text(
        "\n".join(
            [
                "<expr>ts_mean(crypto.returns(5))</expr>",
                "<expr>future_return(crypto.returns(5))</expr>",
                "<expr>ts_mean(crsp.dlyret(5))</expr>",
            ]
        )
    )

    out = generate_oracle_seed_candidates_from_file(path, namespace="crypto")

    assert out == ("ts_mean(crypto.returns(5))",)


def test_generate_oracle_seed_candidates_calls_openrouter(monkeypatch):
    scenario = FactorScenario.from_benchmark(
        "single_asset_direction",
        market="crypto",
        horizon=1,
    )
    payload = {
        "choices": [
            {
                "message": {
                    "content": "<expr>sign(diff(crypto.close(2)))</expr>\n<expr>ts_mean(crsp.dlyret(5))</expr>"
                }
            }
        ]
    }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    with patch("src.llm.openrouter.load_dotenv"):
        with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)) as urlopen:
            out = generate_oracle_seed_candidates(
                scenario,
                namespace="crypto",
                count=2,
                config=OpenRouterConfig(model="test/model", timeout_seconds=1),
            )

    req = urlopen.call_args.args[0]
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "test/model"
    assert out == ("sign(diff(crypto.close(2)))",)


def test_model_role_defaults_match_roles_config():
    import yaml
    from src.rft import MinerConfig

    roles_path = Path("config/openrouter_llm_roles.yaml")
    if roles_path.exists():
        roles = yaml.safe_load(roles_path.read_text())
        assert OpenRouterConfig().model == roles["oracle_llm"]["default_model"]
        assert MinerConfig().model == roles["miner_llm"]["default_model"]

