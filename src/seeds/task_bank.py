"""Seeded task-bank construction."""

from __future__ import annotations

from dataclasses import dataclass

from src.seeds.pool import SeedCandidate
from src.seeds.scenario import FactorScenario


@dataclass(frozen=True)
class EvaluationWindow:
    """A market interval used to evaluate generated factors."""

    start: str
    end: str


@dataclass(frozen=True)
class SeedTask:
    """One seeded factor-mining task."""

    seed_expr: str
    seed_score: float
    scenario: FactorScenario
    window: EvaluationWindow
    objective: str


def build_task_bank(
    seeds: tuple[SeedCandidate, ...] | list[SeedCandidate],
    scenario: FactorScenario,
    windows: tuple[EvaluationWindow, ...] | list[EvaluationWindow],
) -> tuple[SeedTask, ...]:
    """Construct the Cartesian product of seed pool and evaluation windows."""

    return tuple(
        SeedTask(
            seed_expr=seed.expr,
            seed_score=seed.score,
            scenario=scenario,
            window=window,
            objective=scenario.objective,
        )
        for seed in seeds
        for window in windows
    )
