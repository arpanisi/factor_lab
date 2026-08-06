"""Check OpenRouter token usage for Factor Lab prompts."""

from __future__ import annotations

import argparse

from src.llm.openrouter import call_openrouter_chat, openrouter_usage
from src.rft.prompts import build_miner_messages
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank, refine_scenario
from src.seeds.oracle import build_oracle_seed_messages


def build_messages(
    *,
    prompt_kind: str,
    raw_scenario: str,
    namespace: str | None,
    count: int,
    seed_expr: str,
    seed_score: float,
) -> list[dict[str, str]]:
    """Build the requested Factor Lab messages."""

    refined = refine_scenario(raw_scenario)
    ns = namespace or refined.namespace
    if prompt_kind == "oracle":
        return build_oracle_seed_messages(refined.scenario, namespace=ns, count=count)

    task = build_task_bank(
        [SeedCandidate(seed_expr, float(seed_score), "")],
        refined.scenario,
        [EvaluationWindow("2020-01-01", "2020-12-31")],
    )[0]
    return build_miner_messages(task, namespace=ns, count=count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenRouter token usage for Factor Lab prompts.")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--prompt-kind", choices=("oracle", "miner"), default="oracle")
    parser.add_argument("--raw-scenario", default="Find crypto cross-sectional RankIC factors with 1d horizon")
    parser.add_argument("--namespace")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed-expr", default="div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))")
    parser.add_argument("--seed-score", type=float, default=0.655671862964597)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    messages = build_messages(
        prompt_kind=args.prompt_kind,
        raw_scenario=args.raw_scenario,
        namespace=args.namespace,
        count=args.count,
        seed_expr=args.seed_expr,
        seed_score=args.seed_score,
    )
    data = call_openrouter_chat(
        model=args.model,
        messages=messages,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    usage = openrouter_usage(data)
    char_count = sum(len(message["content"]) for message in messages)
    print(f"model: {args.model}")
    print(f"prompt_kind: {args.prompt_kind}")
    print(f"message_chars: {char_count}")
    print(f"prompt_tokens: {usage.prompt_tokens}")
    print(f"completion_tokens: {usage.completion_tokens}")
    print(f"total_tokens: {usage.total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
