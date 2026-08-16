"""Tier 1 acceptance gate (coding-plan.md Sections 10.4 / 10.5 / 10.6).

Runs the synthetic factor-quality sweep (10.4), the within-rollout-group
reward variance diagnostic (10.5) against the real crypto panel via the real
OpenRouter miner model, and writes the locked Tier 1 JSON report (10.6).

The crypto panel is a hard prerequisite (Section 9.1): if the file is missing,
the gate hard-stops naming the missing path and does not fall back to a
synthetic or empty panel.

Usage (repository root as working directory):

    python -m examples.run_tier1_gate \
        --crypto-panel data/crypto/crypto_panel_clean.pkl \
        --tickers ADA-USD,BNB-USD,BTC-USD,DOGE-USD,ETH-USD,LINK-USD,XLM-USD,XRP-USD \
        --output outputs/tier1/tier1_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from examples.build_seed_bank import crypto_frames_from_panel
from examples.dsl_smoke_test import load_dotenv
from src.data import adapt_crypto_ohlcv, verify_crypto_panel
from src.dsl import ValidationError, validate_expr
from src.llm.openrouter import call_openrouter_chat, openrouter_content
from src.rft import (
    DatabaseSelectionConfig,
    DiCoRewardConfig,
    MinedFactorDatabase,
    build_miner_messages,
    reward_completions,
)
from src.scoring import score_cross_sectional_rankic
from src.seeds import EvaluationWindow, FactorScenario, SeedTask


MIN_HISTORY = 30
LOCKED_CRYPTO_TICKERS = (
    "ADA-USD",
    "BNB-USD",
    "BTC-USD",
    "DOGE-USD",
    "ETH-USD",
    "LINK-USD",
    "XLM-USD",
    "XRP-USD",
)
LOCKED_CRYPTO_TICKERS_ARG = ",".join(LOCKED_CRYPTO_TICKERS)
MIN_ASSETS = 8
MIN_ASSETS_DERIVATION = (
    "Locked minimum universe size of 8 for cross-sectional RankIC; timestamps "
    "with fewer than 8 finite factor/forward-return pairs are skipped, and a "
    "fully undersized evaluation returns score=-1.0."
)
BOOTSTRAP_SEED_EXPR = "div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))"
BOOTSTRAP_SEED_SCORE = 0.38385972330719526
BOOTSTRAP_SEED_DERIVATION = (
    "Section 5.1 cross-sectional RankIC formula against the full locked 8-asset crypto panel history "
    "(ADA-USD, BNB-USD, BTC-USD, DOGE-USD, ETH-USD, LINK-USD, XLM-USD, XRP-USD) with min_assets=8; "
    "recomputed from the locked panel, matching the code-review.md cited value."
)
ORACLE_MODEL = "deepseek/deepseek-chat-v3.1"
MINER_MODEL = "qwen/qwen3-235b-a22b-2507"
ROLLOUT_GROUP_SIZE = 8
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 50}

SWEEP_EXPRESSIONS = (
    "div(ts_sum(crypto.returns(30)), ts_std(crypto.returns(30)))",
    "neg(ts_mean(crypto.returns(5)))",
    "div(ts_mean(crypto.volume(7)), ts_mean(crypto.volume(30)))",
    "ts_mean(crypto.close(1))",
    "div(crypto.close(1), sub(crypto.close(1), crypto.close(1)))",
)
INVALID_EXPRESSION = "ts_mean(ts_mean(crypto.close(1)))"


def run_synthetic_sweep(
    frames: dict[str, pd.DataFrame],
    *,
    min_history: int = MIN_HISTORY,
    min_assets: int = MIN_ASSETS,
) -> dict:
    """Run the Section 10.4 synthetic factor-quality sweep.

    Scores the 5 locked expressions with the Section 5.1 formula directly and
    confirms a genuinely grammar-invalid expression is rejected at validation.
    Returns the ``synthetic_sweep`` section of the locked JSON report.
    """

    validated = []
    for expr in SWEEP_EXPRESSIONS:
        try:
            validate_expr(expr)
            validated.append(True)
        except ValidationError:
            validated.append(False)

    scores: list[float | None] = []
    score_errors: list[str] = []
    for expr in SWEEP_EXPRESSIONS:
        try:
            result = score_cross_sectional_rankic(
                expr,
                "crypto",
                frames,
                price_col="close",
                min_history=min_history,
                min_assets=min_assets,
            )
            scores.append(float(result.score))
            score_errors.append("")
        except Exception as exc:  # noqa: BLE001 - report any scoring failure verbatim
            scores.append(None)
            score_errors.append(str(exc))

    correctly_rejected = False
    rejection_error = ""
    try:
        validate_expr(INVALID_EXPRESSION)
    except ValidationError as exc:
        correctly_rejected = True
        rejection_error = str(exc)

    top_three = [scores[0], scores[1], scores[2]]
    top_three_finite_in_range = bool(
        all(s is not None and math.isfinite(s) and -1.0 <= s <= 1.0 for s in top_three)
    )
    top_three_not_identical = bool(
        all(s is not None for s in top_three) and len(set(top_three)) > 1  # type: ignore[arg-type]
    )
    expression_4_finite = scores[3] is not None and math.isfinite(scores[3])  # type: ignore[arg-type]
    expression_4_distinct_from_1_3 = bool(
        expression_4_finite and all(scores[3] != s for s in top_three if s is not None)
    )
    expression_5_degenerate = scores[4] == -1.0

    undersized_frames = {k: v for i, (k, v) in enumerate(frames.items()) if i < max(1, min_assets - 1)}
    undersized_score: float | None = None
    try:
        undersized_result = score_cross_sectional_rankic(
            SWEEP_EXPRESSIONS[0],
            "crypto",
            undersized_frames,
            price_col="close",
            min_history=min_history,
            min_assets=min_assets,
        )
        undersized_score = float(undersized_result.score)
    except Exception:  # noqa: BLE001
        undersized_score = None
    undersized_guard_passed = bool(undersized_score == -1.0)

    criteria = {
        "all_sweep_expressions_validate": bool(all(validated)),
        "invalid_expression_rejected_at_validation": bool(correctly_rejected),
        "expressions_1_3_finite_in_range": top_three_finite_in_range,
        "expressions_1_3_not_identical": top_three_not_identical,
        "expression_4_finite": expression_4_finite,
        "expression_4_distinct_from_1_3": expression_4_distinct_from_1_3,
        "expression_5_degenerate_score_minus_one": expression_5_degenerate,
        "undersized_universe_guard_returns_minus_one": undersized_guard_passed,
    }

    for name, holds in criteria.items():
        print(f"[tier1-gate] sweep criterion {name}: {'PASS' if holds else 'FAIL'}")
    if any(score_errors):
        for expr, err in zip(SWEEP_EXPRESSIONS, score_errors):
            if err:
                print(f"[tier1-gate] sweep scoring error for {expr}: {err}")

    return {
        "expressions": list(SWEEP_EXPRESSIONS),
        "scores": [s if s is not None else None for s in scores],
        "invalid_expression_rejected": {
            "expr": INVALID_EXPRESSION,
            "correctly_rejected": bool(correctly_rejected),
            "error": rejection_error,
        },
        "undersized_universe_guard": {
            "tested_asset_count": len(undersized_frames),
            "min_assets": min_assets,
            "score": undersized_score,
            "correctly_rejected": undersized_guard_passed,
        },
        "passed": bool(all(criteria.values())),
    }


def _build_windows(panel: dict, *, n: int, min_history: int) -> tuple[EvaluationWindow, ...]:
    """Split the panel date range into ``n`` contiguous, distinct windows.

    Each window is required to hold at least ``min_history + 1`` bars so the
    Section 5.1 scorers always have at least one scored timestamp.
    """

    index = panel["close"].index
    if len(index) < n:
        raise RuntimeError(
            f"crypto panel has {len(index)} bars; cannot build {n} distinct evaluation windows"
        )
    edges = [int(round(i * len(index) / n)) for i in range(n + 1)]
    windows = []
    for i in range(n):
        chunk = index[edges[i]:edges[i + 1]]
        if len(chunk) < min_history + 1:
            raise RuntimeError(
                f"evaluation window {i} spans only {len(chunk)} bars "
                f"(needs at least {min_history + 1} to score)"
            )
        windows.append(EvaluationWindow(str(chunk[0].date()), str(chunk[-1].date())))
    return tuple(windows)


def _frames_for_window(
    panel: dict,
    tickers: tuple[str, ...],
    window: EvaluationWindow,
) -> dict[str, pd.DataFrame]:
    """Slice the crypto panel to one evaluation window into DSL-ready frames."""

    start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
    frames = {}
    for ticker in tickers:
        frame = pd.DataFrame(
            {key: panel[key].loc[start:end][ticker] for key in ("open", "high", "low", "close", "volume")}
        )
        if "returns" in panel:
            frame["returns"] = panel["returns"].loc[start:end][ticker]
        frames[str(ticker)] = adapt_crypto_ohlcv(frame)
    return frames


def build_tier1_tasks(
    panel: dict,
    tickers: tuple[str, ...],
    *,
    min_history: int = MIN_HISTORY,
) -> tuple[tuple[SeedTask, object, str], ...]:
    """Build the fixed 5-task set: 3 cross-sectional + 2 directional (Section 10.5).

    Every task carries the locked bootstrap seed against a distinct evaluation
    window. The second element is the window-sliced data the scorer runs on;
    the third is the price column.
    """

    windows = _build_windows(panel, n=5, min_history=min_history)
    cross_scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="tier1_crypto_cross_sectional_rankic",
    )
    direction_scenario = FactorScenario.from_benchmark(
        "single_asset_direction",
        market="crypto",
        horizon=1,
        name="tier1_crypto_single_asset_direction",
    )

    tasks = []
    for i, window in enumerate(windows):
        if i < 3:
            scenario = cross_scenario
            data = _frames_for_window(panel, tickers, window)
        else:
            scenario = direction_scenario
            data = _frames_for_window(panel, tickers, window)[tickers[0]]
        tasks.append(
            (
                SeedTask(
                    seed_expr=BOOTSTRAP_SEED_EXPR,
                    seed_score=BOOTSTRAP_SEED_SCORE,
                    scenario=scenario,
                    window=window,
                    objective=scenario.objective,
                ),
                data,
                "close",
            )
        )
    return tuple(tasks)


def _rollout_record(result) -> dict:
    """Convert one RewardBridgeResult into the locked per-rollout record."""

    score = float(result.score) if math.isfinite(result.score) else -1.0
    reward = float(result.reward) if math.isfinite(result.reward) else -1.0
    return {
        "completion": result.completion,
        "expr": result.expr,
        "valid": bool(result.valid),
        "score": score,
        "reward": reward,
        "reason": result.reason,
    }


def run_reward_variance_diagnostic(
    tasks: tuple[tuple[SeedTask, object, str], ...],
    *,
    miner_model: str = MINER_MODEL,
    min_history: int = MIN_HISTORY,
    min_assets: int = MIN_ASSETS,
    k: int = ROLLOUT_GROUP_SIZE,
) -> dict:
    """Run the Section 10.5 within-rollout-group reward variance diagnostic.

    K = ``k`` independent completions per task from the real OpenRouter-backed
    miner model at the locked sampling parameters; each completion is scored
    and rewarded with the real reward pipeline end to end (reward bridge).
    """

    archive = MinedFactorDatabase()
    selection_config = DatabaseSelectionConfig()
    reward_config = DiCoRewardConfig()

    task_records = []
    for task_id, (task, data, price_col) in enumerate(tasks):
        messages = build_miner_messages(task, namespace="crypto", count=1)
        completions = []
        for _ in range(k):
            raw = call_openrouter_chat(
                model=miner_model,
                messages=messages,
                temperature=SAMPLING["temperature"],
                max_tokens=1600,
                top_p=SAMPLING["top_p"],
                top_k=SAMPLING["top_k"],
            )
            completions.append(openrouter_content(raw))

        results = reward_completions(
            completions,
            task=task,
            namespace="crypto",
            data=data,
            price_col=price_col,
            archive=archive,
            min_history=min_history,
            min_assets=min_assets,
            reward_config=reward_config,
            selection_config=selection_config,
        )

        rollouts = [_rollout_record(result) for result in results]
        rewards = np.asarray([r["reward"] for r in rollouts], dtype=float)
        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards))

        task_records.append(
            {
                "task_id": task_id,
                "benchmark": task.scenario.benchmark,
                "seed_expr": task.seed_expr,
                "rollouts": rollouts,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
            }
        )
        print(
            f"[tier1-gate] task {task_id} {task.scenario.benchmark} "
            f"window={task.window.start}..{task.window.end} "
            f"reward_mean={reward_mean:.4f} reward_std={reward_std:.4f}"
        )

    tasks_with_std_above_0_05 = sum(1 for record in task_records if record["reward_std"] > 0.05)
    tasks_all_invalid = sum(
        1 for record in task_records if all(r["reward"] == -1.0 for r in record["rollouts"])
    )
    at_least_one_valid_and_scored = any(
        r["valid"] and math.isfinite(r["score"])
        for record in task_records
        for r in record["rollouts"]
    )

    passed = bool(
        at_least_one_valid_and_scored and tasks_with_std_above_0_05 >= 3 and tasks_all_invalid == 0
    )

    return {
        "oracle_model": ORACLE_MODEL,
        "miner_model": miner_model,
        "rollout_group_size_k": k,
        "temperature": SAMPLING["temperature"],
        "top_p": SAMPLING["top_p"],
        "top_k": SAMPLING["top_k"],
        "tasks": task_records,
        "tasks_with_std_above_0_05": int(tasks_with_std_above_0_05),
        "tasks_all_invalid": int(tasks_all_invalid),
        "at_least_one_valid_and_scored": bool(at_least_one_valid_and_scored),
        "passed": passed,
    }


def build_tier1_report(
    *,
    panel: dict,
    panel_path: str,
    tickers: tuple[str, ...],
    min_history: int = MIN_HISTORY,
    miner_model: str = MINER_MODEL,
    k: int = ROLLOUT_GROUP_SIZE,
) -> dict:
    """Run both gate sections against the real panel and assemble the report."""

    min_assets = MIN_ASSETS
    frames = crypto_frames_from_panel(panel, tickers=tickers)

    sweep = run_synthetic_sweep(frames, min_history=min_history, min_assets=min_assets)
    tasks = build_tier1_tasks(panel, tickers, min_history=min_history)
    diagnostic = run_reward_variance_diagnostic(
        tasks,
        miner_model=miner_model,
        min_history=min_history,
        min_assets=min_assets,
        k=k,
    )

    report = {
        "tier": 1,
        "crypto_panel_path": panel_path,
        "tickers": list(tickers),
        "min_history": min_history,
        "min_assets": min_assets,
        "min_assets_derivation": MIN_ASSETS_DERIVATION,
        "bootstrap_seed": {
            "expr": BOOTSTRAP_SEED_EXPR,
            "score": BOOTSTRAP_SEED_SCORE,
            "score_derivation": BOOTSTRAP_SEED_DERIVATION,
        },
        "synthetic_sweep": sweep,
        "reward_variance_diagnostic": diagnostic,
        "overall_passed": bool(sweep["passed"] and diagnostic["passed"]),
    }
    return report


def _hard_stop(message: str) -> None:
    print(f"[tier1-gate] HARD STOP: {message}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Tier 1 acceptance gate (coding-plan.md 10.4/10.5/10.6)."
    )
    parser.add_argument("--crypto-panel", default="data/crypto/crypto_panel_clean.pkl")
    parser.add_argument("--tickers", default=LOCKED_CRYPTO_TICKERS_ARG)
    parser.add_argument("--miner-model", default=MINER_MODEL)
    parser.add_argument("--rollout-group-size", type=int, default=ROLLOUT_GROUP_SIZE)
    parser.add_argument("--min-history", type=int, default=MIN_HISTORY)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "tier1" / "tier1_report.json")
    args = parser.parse_args()

    load_dotenv()

    tickers = tuple(str(item) for item in args.tickers.split(",") if item.strip()) or None
    try:
        panel = verify_crypto_panel(str(args.crypto_panel), tickers=tickers)
    except (FileNotFoundError, ValueError) as exc:
        _hard_stop(
            f"crypto panel prerequisite failed (Section 9.1): {exc}"
        )
        return 1

    if not os.getenv("OPENROUTER_API_KEY"):
        _hard_stop(
            "OPENROUTER_API_KEY is not set in the environment or .env "
            "(Section 9.3 requires it for the Section 10.5 miner rollouts)."
        )
        return 1

    try:
        report = build_tier1_report(
            panel=panel,
            panel_path=str(args.crypto_panel),
            tickers=tickers or tuple(panel["close"].columns),
            min_history=args.min_history,
            miner_model=args.miner_model,
            k=args.rollout_group_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _hard_stop(str(exc))
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    sweep = report["synthetic_sweep"]
    diag = report["reward_variance_diagnostic"]
    print(
        "[tier1-gate] synthetic sweep passed={} | "
        "tasks_with_std_above_0_05={}/{} tasks_all_invalid={} "
        "at_least_one_valid_and_scored={} | overall_passed={}".format(
            sweep["passed"],
            diag["tasks_with_std_above_0_05"],
            len(diag["tasks"]),
            diag["tasks_all_invalid"],
            diag["at_least_one_valid_and_scored"],
            report["overall_passed"],
        )
    )
    print(f"[tier1-gate] report written to {args.output}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
