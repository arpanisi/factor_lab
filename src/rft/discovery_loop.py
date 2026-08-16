"""Local miner-baseline discovery loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd

from src.rft.miner import MinerConfig, generate_miner_candidates
from src.rft.realizer import RealizedFactor, realize_factor
from src.scoring import CrossSectionalScore, TimeSeriesScore, score_cross_sectional_rankic, score_directional_prediction
from src.seeds.task_bank import SeedTask


CandidateGenerator = Callable[[SeedTask, str, int], tuple[str, ...]]


@dataclass(frozen=True)
class DiscoveryResult:
    """Evaluated generated candidate."""

    expr: str
    valid: bool
    signature: str
    score: float
    metrics: dict[str, float]
    reward: float | None = None
    reward_components: dict[str, float] | None = None
    stored: bool = False
    reason: str = ""


def run_discovery_for_task(
    task: SeedTask,
    *,
    namespace: str,
    data,
    price_col: str,
    count: int = 4,
    min_history: int = 1,
    min_assets: int = 8,
    generator: CandidateGenerator | None = None,
    miner_config: MinerConfig | None = None,
    archive=None,
    reward_config=None,
    selection_config=None,
) -> tuple[DiscoveryResult, ...]:
    """Generate, realize, and empirically evaluate candidates for one task."""

    candidates = (
        generator(task, namespace, count)
        if generator is not None
        else generate_miner_candidates(task, namespace=namespace, count=count, config=miner_config)
    )
    results = []
    seen = set()
    for expr in candidates:
        realized = realize_factor(expr)
        if not realized.valid:
            results.append(_invalid_result(expr, realized))
            continue
        if realized.signature in seen:
            continue
        seen.add(realized.signature)

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
            results.append(
                DiscoveryResult(
                    expr=realized.expr,
                    valid=False,
                    signature=realized.signature,
                    score=-1.0,
                    metrics={},
                    reason=str(exc),
                )
            )
            continue

        reward_value = None
        reward_components = None
        stored = False
        if archive is not None:
            from src.rft.dico_reward import compute_dico_reward
            from src.rft.prompts import task_to_prompt_payload

            candidate = DiscoveryResult(
                expr=realized.expr,
                valid=True,
                signature=realized.signature,
                score=score,
                metrics=metrics,
            )
            reward = compute_dico_reward(candidate, archive, config=reward_config)
            reward_value = reward.reward
            reward_components = {
                "quality": reward.quality,
                "diversity": reward.diversity,
                "complementarity": reward.complementarity,
                "duplicate": float(reward.duplicate),
            }
            stored = archive.maybe_add(
                candidate,
                reward,
                task_payload=task_to_prompt_payload(task, namespace=namespace),
                config=selection_config,
            )

        results.append(
            DiscoveryResult(
                expr=realized.expr,
                valid=True,
                signature=realized.signature,
                score=score,
                metrics=metrics,
                reward=reward_value,
                reward_components=reward_components,
                stored=stored,
            )
        )

    return tuple(results)


def _score_realized(
    expr: str,
    task: SeedTask,
    *,
    namespace: str,
    data,
    price_col: str,
    min_history: int,
    min_assets: int,
) -> tuple[float, dict[str, float]]:
    if task.scenario.benchmark == "daily_cross_sectional_rankic":
        if not isinstance(data, Mapping):
            raise TypeError("cross-sectional task requires data as Mapping[asset, DataFrame]")
        result = score_cross_sectional_rankic(
            expr,
            namespace,
            data,
            price_col=price_col,
            horizon=task.scenario.horizon,
            min_history=min_history,
            min_assets=min_assets,
        )
        return result.score, _cross_sectional_metrics(result)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("directional task requires data as a single DataFrame")
    result = score_directional_prediction(
        expr,
        namespace,
        data,
        price_col=price_col,
        horizon=task.scenario.horizon,
        min_history=min_history,
    )
    return result.score, _time_series_metrics(result)


def _invalid_result(expr: str, realized: RealizedFactor) -> DiscoveryResult:
    return DiscoveryResult(
        expr=expr,
        valid=False,
        signature=realized.signature,
        score=-1.0,
        metrics={},
        reward=None,
        reward_components=None,
        stored=False,
        reason=realized.reason,
    )


def _cross_sectional_metrics(result: CrossSectionalScore) -> dict[str, float]:
    return {
        "score": result.score,
        "rank_ic_mean": result.rank_ic_mean,
        "rank_ic_std": result.rank_ic_std,
        "rank_ic_ir": result.rank_ic_ir,
        "long_short_mean": result.long_short_mean,
        "long_short_sharpe": result.long_short_sharpe,
        "valid_times": float(result.valid_times),
        "mean_assets_per_time": result.mean_assets_per_time,
    }


def _time_series_metrics(result: TimeSeriesScore) -> dict[str, float]:
    return {
        "score": result.score,
        "directional_accuracy": result.directional_accuracy,
        "ic": result.ic,
        "rank_ic": result.rank_ic,
        "samples": float(result.samples),
    }
