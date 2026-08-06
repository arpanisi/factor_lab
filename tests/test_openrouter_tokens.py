from examples.check_openrouter_tokens import build_messages
from src.llm.openrouter import openrouter_content, openrouter_usage


def test_openrouter_usage_parses_token_counts():
    usage = openrouter_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 2
    assert usage.total_tokens == 12


def test_openrouter_content_extracts_message_content():
    content = openrouter_content({"choices": [{"message": {"content": "hello"}}]})

    assert content == "hello"


def test_token_check_builds_oracle_messages():
    messages = build_messages(
        prompt_kind="oracle",
        raw_scenario="Find crypto cross-sectional RankIC factors",
        namespace=None,
        count=3,
        seed_expr="ts_mean(crypto.returns(2))",
        seed_score=0.1,
    )

    assert messages[0]["role"] == "system"
    assert "Generate 3" in messages[1]["content"]
    assert "crypto.returns" in messages[1]["content"]


def test_token_check_builds_miner_messages():
    messages = build_messages(
        prompt_kind="miner",
        raw_scenario="Find crypto cross-sectional RankIC factors",
        namespace=None,
        count=2,
        seed_expr="ts_mean(crypto.returns(2))",
        seed_score=0.1,
    )

    assert "Seed expression" in messages[1]["content"]
    assert "Generate 2" in messages[1]["content"]
