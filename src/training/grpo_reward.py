"""TRL/GRPO reward function adapter backed by the Factor DSL oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from src.rft import MinedFactorDatabase
from src.rft.reward_bridge import reward_completion
from src.seeds.task_bank import SeedTask


@dataclass
class FactorRewardRuntime:
    """Runtime objects needed to score generated factor completions."""

    task: SeedTask
    namespace: str
    data: Any
    price_col: str
    archive: MinedFactorDatabase
    min_history: int = 30
    min_assets: int = 3
    reward_log_jsonl: Path | None = None


def make_grpo_reward_func(runtime: FactorRewardRuntime) -> Callable[..., list[float]]:
    """Create a TRL-compatible reward function.

    TRL reward functions receive completions plus optional metadata and return
    one float reward per completion. The scalar reward is the DiCo/backtest score
    already produced by the project reward bridge.
    """

    def reward_func(completions: Sequence[str], **_: Any) -> list[float]:
        rewards = []
        for completion in completions:
            result = reward_completion(
                str(completion),
                task=runtime.task,
                namespace=runtime.namespace,
                data=runtime.data,
                price_col=runtime.price_col,
                archive=runtime.archive,
                min_history=runtime.min_history,
                min_assets=runtime.min_assets,
            )
            if runtime.reward_log_jsonl is not None:
                _append_reward_log(runtime.reward_log_jsonl, result)
            rewards.append(float(result.reward))
        return rewards

    return reward_func


def _append_reward_log(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "completion": result.completion,
        "expr": result.expr,
        "reward": float(result.reward),
        "valid": bool(result.valid),
        "score": float(result.score),
        "metrics": result.metrics,
        "stored": bool(result.stored),
        "reason": result.reason,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
