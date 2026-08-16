"""OpenRouter-backed local miner baseline."""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.openrouter import call_openrouter_chat, openrouter_content
from src.rft.prompts import build_miner_messages
from src.rft.realizer import extract_factor_expressions
from src.seeds.task_bank import SeedTask


@dataclass(frozen=True)
class MinerConfig:
    """OpenRouter request settings for the local miner baseline."""

    model: str = "qwen/qwen3-235b-a22b-2507"
    temperature: float = 0.8
    max_tokens: int = 1600
    timeout_seconds: int = 60


def generate_miner_candidates(
    task: SeedTask,
    *,
    namespace: str,
    count: int,
    config: MinerConfig | None = None,
    api_key: str | None = None,
) -> tuple[str, ...]:
    """Generate candidate factor expressions for one seeded task."""

    cfg = config or MinerConfig()
    data = call_openrouter_chat(
        model=cfg.model,
        messages=build_miner_messages(task, namespace=namespace, count=count),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout_seconds=cfg.timeout_seconds,
        api_key=api_key,
    )
    content = openrouter_content(data)
    return extract_factor_expressions(content)
