"""CLI for creating a Verl GRPO prompt parquet from an existing seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.build_seed_bank import crypto_frames_from_panel
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL
from src.data import verify_crypto_panel
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank
from src.verl_integration.dataset import build_verl_prompt_rows, write_verl_prompt_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Factor Lab prompt parquet for Verl GRPO.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-expr", required=True)
    parser.add_argument("--seed-score", type=float, default=0.0)
    parser.add_argument("--namespace", default="crypto")
    parser.add_argument("--market", default="crypto")
    parser.add_argument("--benchmark", default="daily_cross_sectional_rankic")
    parser.add_argument("--scenario-name", default="crypto_verl_factor_discovery")
    parser.add_argument("--window-start")
    parser.add_argument("--window-end")
    parser.add_argument("--crypto-panel", type=Path, default=DEFAULT_CRYPTO_PANEL)
    parser.add_argument("--tickers", default="ADA-USD,BNB-USD,BTC-USD,DOGE-USD,ETH-USD,LINK-USD,XLM-USD,XRP-USD")
    parser.add_argument("--repeats", type=int, default=400)
    args = parser.parse_args()

    start = args.window_start
    end = args.window_end
    if args.namespace == "crypto" and (start is None or end is None):
        tickers = tuple(item.strip() for item in args.tickers.split(",") if item.strip()) or None
        panel = verify_crypto_panel(args.crypto_panel, tickers=tickers)
        frames = crypto_frames_from_panel(panel, tickers=tickers)
        if not frames:
            raise ValueError("no crypto frames were loaded")
        close = panel["close"]
        start = start or str(close.index.min().date())
        end = end or str(close.index.max().date())
    if start is None or end is None:
        raise ValueError("--window-start and --window-end are required for non-crypto datasets")

    scenario = FactorScenario.from_benchmark(
        args.benchmark,
        market=args.market,
        horizon=1,
        name=args.scenario_name,
    )
    tasks = build_task_bank(
        [SeedCandidate(args.seed_expr, float(args.seed_score), "")],
        scenario,
        [EvaluationWindow(start, end)],
    )
    rows = build_verl_prompt_rows(tasks, namespace=args.namespace, repeats=args.repeats)
    write_verl_prompt_dataset(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
