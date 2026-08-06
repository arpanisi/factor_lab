"""Proxy implementations of compared LLM alpha-discovery approaches.

The paper compares against named external systems. This module keeps the
comparison fair inside this project by using the same DSL, variables, operators,
and post-selection protocol while varying the candidate-generation workflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from src.evaluation import PostSelectionConfig, evaluate_factor_library
from src.evaluation.post_selection import FusedEvaluation, write_evaluation_report
from examples.build_seed_bank import crypto_frames_from_panel
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL
from src.rft import MinerConfig
from src.rft.miner import generate_miner_candidates
from src.seeds import EvaluationWindow, FactorScenario, SeedPoolConfig, build_scenario_seed_bank
from src.seeds.candidates import candidate_templates_for_scenario
from src.seeds.oracle import OpenRouterConfig, generate_oracle_seed_candidates


DEFAULT_BENCHMARK_OUTPUT_DIR = Path("factor_lab") / "outputs" / "compared_approaches"


@dataclass(frozen=True)
class BenchmarkApproach:
    """One compared approach and how it generates candidate factors."""

    name: str
    description: str


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Candidate and evaluation output for one compared approach."""

    approach: str
    raw_candidate_count: int
    valid_candidate_count: int
    selected_exprs: tuple[str, ...]
    evaluation: FusedEvaluation


APPROACHES = {
    "alphabench": BenchmarkApproach(
        "alphabench",
        "Prompt-based one-shot formulaic alpha generation baseline.",
    ),
    "quantaalpha": BenchmarkApproach(
        "quantaalpha",
        "Evolutionary LLM mutation/crossover-style candidate generation baseline.",
    ),
    "rd_agent": BenchmarkApproach(
        "rd_agent",
        "Researcher/developer iterative proposal and refinement baseline.",
    ),
    "alpha_jungle": BenchmarkApproach(
        "alpha_jungle",
        "MCTS-style symbolic exploration baseline with seed expansion.",
    ),
    "factor_lab": BenchmarkApproach(
        "factor_lab",
        "This project's oracle-seed plus miner-rollout workflow.",
    ),
}


CandidateGenerator = Callable[[FactorScenario, str, Mapping[str, pd.DataFrame], str, str, int], tuple[str, ...]]


def run_compared_approaches(
    *,
    model: str,
    panel_path: Path = DEFAULT_CRYPTO_PANEL,
    tickers: tuple[str, ...] | None = None,
    approach_names: tuple[str, ...] = ("alphabench", "quantaalpha", "rd_agent", "alpha_jungle", "factor_lab"),
    count: int = 12,
    top_k: int = 5,
    correlation_threshold: float = 0.7,
    output_dir: Path = DEFAULT_BENCHMARK_OUTPUT_DIR,
    generator: CandidateGenerator | None = None,
) -> dict:
    """Run compared approaches under one DSL/evaluation protocol."""

    panel = pd.read_pickle(panel_path)
    frames = crypto_frames_from_panel(panel, tickers=tickers)
    dates = pd.DatetimeIndex(panel["close"].index).sort_values()
    split_idx = int(len(dates) * 0.7)
    split = pd.Timestamp(dates[split_idx])
    validation_start = str(dates.min().date())
    validation_end = str(split.date())
    test_start = str(pd.Timestamp(dates[min(split_idx + 1, len(dates) - 1)]).date())
    test_end = str(dates.max().date())
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="crypto_compared_approaches",
    )
    windows = [EvaluationWindow(validation_start, validation_end)]
    cfg = PostSelectionConfig(
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
        min_history=30,
        min_assets=max(3, min(5, len(frames))),
        horizon=1,
    )

    gen = generator or _generate_candidates_for_approach
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for approach in approach_names:
        if approach not in APPROACHES:
            raise ValueError(f"unknown benchmark approach: {approach}")
        candidates = gen(scenario, approach, frames, "close", model, count)
        seed_bank = build_scenario_seed_bank(
            scenario,
            namespace="crypto",
            data=frames,
            price_col="close",
            windows=windows,
            raw_candidates=candidates,
            min_history=30,
            min_assets=max(3, min(5, len(frames))),
            pool_config=SeedPoolConfig(top_k=max(top_k, 1), quality_threshold=-1.0),
        )
        exprs = tuple(seed.expr for seed in seed_bank.seeds)
        evaluation = evaluate_factor_library(exprs, "crypto", frames, price_col="close", config=cfg)
        result = BenchmarkRunResult(
            approach=approach,
            raw_candidate_count=len(candidates),
            valid_candidate_count=len(exprs),
            selected_exprs=evaluation.selected_exprs,
            evaluation=evaluation,
        )
        write_evaluation_report(output_dir / f"{approach}_evaluation.json", evaluation)
        results.append(result)

    summary = {
        "model": model,
        "count_per_approach": count,
        "correlation_threshold": correlation_threshold,
        "top_k": top_k,
        "results": [_summary_row(result) for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {"results": results, "summary": summary, "output_dir": output_dir}


def _generate_candidates_for_approach(
    scenario: FactorScenario,
    approach: str,
    frames: Mapping[str, pd.DataFrame],
    price_col: str,
    model: str,
    count: int,
) -> tuple[str, ...]:
    if approach == "alphabench":
        return generate_oracle_seed_candidates(
            scenario,
            namespace="crypto",
            count=count,
            config=OpenRouterConfig(model=model, temperature=0.7),
        )
    if approach == "factor_lab":
        seeds = generate_oracle_seed_candidates(
            scenario,
            namespace="crypto",
            count=max(count, 8),
            config=OpenRouterConfig(model=model, temperature=0.7),
        )
        seed_bank = build_scenario_seed_bank(
            scenario,
            namespace="crypto",
            data=frames,
            price_col=price_col,
            windows=[EvaluationWindow("", "")],
            raw_candidates=seeds,
            min_history=30,
            min_assets=max(3, min(5, len(frames))),
            pool_config=SeedPoolConfig(top_k=3, quality_threshold=-1.0),
        )
        candidates = []
        for task in seed_bank.tasks:
            candidates.extend(
                generate_miner_candidates(
                    task,
                    namespace="crypto",
                    count=max(1, count // max(1, len(seed_bank.tasks))),
                    config=MinerConfig(model=model, temperature=0.8),
                )
            )
        return tuple(candidates)

    templates = tuple(expr for expr in candidate_templates_for_scenario(scenario) if "crypto." in expr)
    if approach == "quantaalpha":
        return _template_mutations(templates, count)
    if approach == "rd_agent":
        return _rd_agent_refinements(templates, count)
    if approach == "alpha_jungle":
        return _mcts_style_expansions(templates, count)
    raise ValueError(f"unknown benchmark approach: {approach}")


def _template_mutations(templates: tuple[str, ...], count: int) -> tuple[str, ...]:
    windows = (5, 7, 10, 14, 20, 30, 45, 60)
    out = []
    for expr in templates:
        for old in ("7", "20", "30"):
            for new in windows:
                out.append(expr.replace(f"({old})", f"({new})"))
                if len(out) >= count:
                    return tuple(out)
    return tuple(out[:count])


def _rd_agent_refinements(templates: tuple[str, ...], count: int) -> tuple[str, ...]:
    wrappers = ("tanh({expr})", "zscore({expr})", "neg({expr})", "abs({expr})")
    out = []
    for expr in templates:
        out.append(expr)
        for wrapper in wrappers:
            out.append(wrapper.format(expr=expr))
            if len(out) >= count:
                return tuple(out)
    return tuple(out[:count])


def _mcts_style_expansions(templates: tuple[str, ...], count: int) -> tuple[str, ...]:
    atoms = (
        "ts_mean(crypto.returns(7))",
        "ts_std(crypto.returns(30))",
        "ts_mean(crypto.volume(20))",
        "ts_mean(crypto.close(14))",
    )
    out = list(templates)
    for left in atoms:
        for right in atoms:
            if left == right:
                continue
            out.append(f"div({left}, {right})")
            out.append(f"sub({left}, {right})")
            if len(out) >= count:
                return tuple(out[:count])
    return tuple(out[:count])


def _summary_row(result: BenchmarkRunResult) -> dict:
    return {
        "approach": result.approach,
        "raw_candidate_count": result.raw_candidate_count,
        "valid_candidate_count": result.valid_candidate_count,
        "selected_count": len(result.selected_exprs),
        "validation": asdict(result.evaluation.validation),
        "test": asdict(result.evaluation.test),
    }
