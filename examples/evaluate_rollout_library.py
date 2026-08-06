"""Evaluate saved baseline rollouts with paper-style post-selection and fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation import PostSelectionConfig, evaluate_factor_library
from src.evaluation.post_selection import write_evaluation_report
from examples.baseline_rollout import DEFAULT_OUTPUT_DIR
from examples.build_seed_bank import crypto_frames_from_panel
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL


def load_valid_rollout_exprs(path: Path) -> list[str]:
    """Load valid rollout expressions from a baseline_rollout.json file."""

    payload = json.loads(path.read_text())
    exprs = []
    seen = set()
    for row in payload.get("rollouts", []):
        expr = str(row.get("expr", "")).strip()
        if not row.get("valid") or not expr or expr in seen:
            continue
        seen.add(expr)
        exprs.append(expr)
    return exprs


def run_crypto_rollout_library_evaluation(
    *,
    rollout_json: Path = DEFAULT_OUTPUT_DIR / "baseline_rollout.json",
    panel_path: Path = DEFAULT_CRYPTO_PANEL,
    output_path: Path = DEFAULT_OUTPUT_DIR / "paper_style_evaluation.json",
    tickers: tuple[str, ...] | None = None,
    validation_start: str | None = None,
    validation_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    top_k: int = 5,
    correlation_threshold: float = 0.7,
) -> dict:
    """Run validation-guided decorrelated selection and equal-weight fusion."""

    exprs = load_valid_rollout_exprs(rollout_json)
    panel = pd.read_pickle(panel_path)
    frames = crypto_frames_from_panel(panel, tickers=tickers)
    if validation_start is None or validation_end is None or test_start is None or test_end is None:
        dates = pd.DatetimeIndex(panel["close"].index).sort_values()
        split = pd.Timestamp(dates[int(len(dates) * 0.7)])
        validation_start = str(dates.min().date())
        validation_end = str(split.date())
        test_start = str((split + pd.Timedelta(days=1)).date())
        test_end = str(dates.max().date())

    result = evaluate_factor_library(
        exprs,
        "crypto",
        frames,
        price_col="close",
        config=PostSelectionConfig(
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
            correlation_threshold=correlation_threshold,
            top_k=top_k,
            min_history=30,
            min_assets=max(3, min(5, len(frames))),
            horizon=1,
        ),
    )
    write_evaluation_report(output_path, result)
    return {"evaluation": result, "output_path": output_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved rollouts with paper-style post-selection.")
    parser.add_argument("--rollout-json", type=Path, default=DEFAULT_OUTPUT_DIR / "baseline_rollout.json")
    parser.add_argument("--crypto-panel", type=Path, default=DEFAULT_CRYPTO_PANEL)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_DIR / "paper_style_evaluation.json")
    parser.add_argument("--tickers", default="BTC-USD,ETH-USD,XRP-USD")
    parser.add_argument("--validation-start")
    parser.add_argument("--validation-end")
    parser.add_argument("--test-start")
    parser.add_argument("--test-end")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--correlation-threshold", type=float, default=0.7)
    args = parser.parse_args()

    tickers = tuple(item.strip() for item in args.tickers.split(",") if item.strip()) or None
    result = run_crypto_rollout_library_evaluation(
        rollout_json=args.rollout_json,
        panel_path=args.crypto_panel,
        output_path=args.output_path,
        tickers=tickers,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
        top_k=args.top_k,
        correlation_threshold=args.correlation_threshold,
    )
    evaluation = result["evaluation"]
    print(
        "paper-style evaluation: "
        f"selected={len(evaluation.selected_exprs)}, "
        f"validation_rankic={evaluation.validation.rank_ic_mean:.6f}, "
        f"test_rankic={evaluation.test.rank_ic_mean:.6f}, "
        f"test_icir={evaluation.test.icir:.6f}, "
        f"test_ls_sharpe={evaluation.test.long_short_sharpe:.6f}"
    )
    print(f"json={result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
