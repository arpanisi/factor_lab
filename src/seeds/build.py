"""Scenario-level seed construction with empirical scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.dsl.parser import CallNode, ExprNode, FieldNode, NumberNode, parse_expr
from src.scoring import score_cross_sectional_rankic, score_directional_prediction
from src.seeds.candidates import candidate_templates_for_scenario
from src.seeds.pool import SeedCandidate, SeedPoolConfig, build_seed_pool
from src.seeds.scenario import FactorScenario
from src.seeds.task_bank import EvaluationWindow, SeedTask, build_task_bank


@dataclass(frozen=True)
class ScenarioSeedBuild:
    """Output from scenario-level seed construction."""

    scenario: FactorScenario
    namespace: str
    seeds: tuple[SeedCandidate, ...]
    tasks: tuple[SeedTask, ...]
    raw_candidate_count: int
    scored_candidate_count: int


def build_scenario_seed_bank(
    scenario: FactorScenario,
    *,
    namespace: str,
    data,
    price_col: str,
    windows: tuple[EvaluationWindow, ...] | list[EvaluationWindow],
    raw_candidates: tuple[str, ...] | list[str] | None = None,
    min_history: int = 1,
    min_assets: int = 3,
    pool_config: SeedPoolConfig | None = None,
) -> ScenarioSeedBuild:
    """Build empirically scored seeds and seed-window tasks for a scenario."""

    raw_candidates = tuple(raw_candidates) if raw_candidates is not None else candidate_templates_for_scenario(scenario)
    namespace_candidates = tuple(expr for expr in raw_candidates if _expr_uses_only_namespace(expr, namespace))

    score_map = {
        expr: _score_candidate(
            expr,
            scenario,
            namespace=namespace,
            data=data,
            price_col=price_col,
            min_history=min_history,
            min_assets=min_assets,
        )
        for expr in namespace_candidates
    }
    seeds = build_seed_pool(namespace_candidates, scores=score_map, config=pool_config)
    tasks = build_task_bank(seeds, scenario, windows)

    return ScenarioSeedBuild(
        scenario=scenario,
        namespace=namespace,
        seeds=seeds,
        tasks=tasks,
        raw_candidate_count=len(raw_candidates),
        scored_candidate_count=len(namespace_candidates),
    )


def _score_candidate(
    expr: str,
    scenario: FactorScenario,
    *,
    namespace: str,
    data,
    price_col: str,
    min_history: int,
    min_assets: int,
) -> float:
    if scenario.benchmark == "daily_cross_sectional_rankic":
        if not isinstance(data, Mapping):
            raise TypeError("cross-sectional scenarios require data as Mapping[asset, DataFrame]")
        return score_cross_sectional_rankic(
            expr,
            namespace,
            data,
            price_col=price_col,
            horizon=scenario.horizon,
            min_history=min_history,
            min_assets=min_assets,
        ).score

    if not isinstance(data, pd.DataFrame):
        raise TypeError("directional scenarios require data as a single DataFrame")
    return score_directional_prediction(
        expr,
        namespace,
        data,
        price_col=price_col,
        horizon=scenario.horizon,
        min_history=min_history,
    ).score


def _expr_uses_only_namespace(expr: str, namespace: str) -> bool:
    try:
        fields = _collect_fields(parse_expr(expr))
    except Exception:
        return False
    return bool(fields) and all(field.namespace == namespace for field in fields)


def _collect_fields(node: ExprNode) -> tuple[FieldNode, ...]:
    if isinstance(node, FieldNode):
        return (node,)
    if isinstance(node, CallNode):
        fields: list[FieldNode] = []
        for arg in node.args:
            fields.extend(_collect_fields(arg))
        return tuple(fields)
    if isinstance(node, NumberNode):
        return ()
    raise TypeError(f"unsupported node type: {type(node).__name__}")
