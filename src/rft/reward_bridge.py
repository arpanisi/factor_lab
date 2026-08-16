"""Reward bridge for generated completions."""

from __future__ import annotations

from dataclasses import dataclass

from src.rft.database import DatabaseSelectionConfig, MinedFactorDatabase
from src.rft.dico_reward import DiCoRewardConfig, compute_dico_reward
from src.rft.discovery_loop import DiscoveryResult, _score_realized
from src.rft.prompts import task_to_prompt_payload
from src.rft.realizer import extract_factor_expressions, realize_factor
from src.seeds.task_bank import SeedTask


@dataclass(frozen=True)
class RewardBridgeResult:
    """Reward output for one completion."""

    completion: str
    expr: str
    reward: float
    valid: bool
    score: float
    metrics: dict[str, float]
    stored: bool
    reason: str = ""


def reward_completion(
    completion: str,
    *,
    task: SeedTask,
    namespace: str,
    data,
    price_col: str,
    archive: MinedFactorDatabase | None = None,
    min_history: int = 1,
    min_assets: int = 8,
    reward_config: DiCoRewardConfig | None = None,
    selection_config: DatabaseSelectionConfig | None = None,
) -> RewardBridgeResult:
    """Convert one generated completion into one scalar reward."""

    expressions = extract_factor_expressions(completion)
    if not expressions:
        return RewardBridgeResult(completion, "", -1.0, False, -1.0, {}, False, "no expression found")

    expr = expressions[0]
    realized = realize_factor(expr)
    if not realized.valid:
        return RewardBridgeResult(completion, expr, -1.0, False, -1.0, {}, False, realized.reason)

    try:
        score, metrics = _score_realized(
            realized.expr,
            task,
            namespace=namespace,
            data=data,
            price_col=price_col,
            min_history=min_history,
            min_assets=min_assets,
        )
    except Exception as exc:
        return RewardBridgeResult(completion, expr, -1.0, False, -1.0, {}, False, str(exc))

    candidate = DiscoveryResult(
        expr=realized.expr,
        valid=True,
        signature=realized.signature,
        score=score,
        metrics=metrics,
    )
    active_archive = archive or MinedFactorDatabase()
    reward_result = compute_dico_reward(candidate, active_archive, config=reward_config)
    stored = False
    if archive is not None:
        stored = archive.maybe_add(
            candidate,
            reward_result,
            task_payload=task_to_prompt_payload(task, namespace=namespace),
            config=selection_config,
        )

    return RewardBridgeResult(
        completion=completion,
        expr=realized.expr,
        reward=reward_result.reward,
        valid=True,
        score=score,
        metrics=metrics,
        stored=stored,
    )


def reward_completions(
    completions: tuple[str, ...] | list[str],
    *,
    task: SeedTask,
    namespace: str,
    data,
    price_col: str,
    archive: MinedFactorDatabase | None = None,
    min_history: int = 1,
    min_assets: int = 8,
    reward_config: DiCoRewardConfig | None = None,
    selection_config: DatabaseSelectionConfig | None = None,
) -> tuple[RewardBridgeResult, ...]:
    """Convert a batch of completions into scalar rewards."""

    return tuple(
        reward_completion(
            completion,
            task=task,
            namespace=namespace,
            data=data,
            price_col=price_col,
            archive=archive,
            min_history=min_history,
            min_assets=min_assets,
            reward_config=reward_config,
            selection_config=selection_config,
        )
        for completion in completions
    )
