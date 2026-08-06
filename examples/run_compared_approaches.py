"""Run compared LLM alpha-discovery approaches under one evaluation protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmarks import run_compared_approaches
from src.benchmarks.compared_approaches import DEFAULT_BENCHMARK_OUTPUT_DIR
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL, load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compared alpha-discovery benchmark approaches.")
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--crypto-panel", type=Path, default=DEFAULT_CRYPTO_PANEL)
    parser.add_argument("--tickers", default="BTC-USD,ETH-USD,XRP-USD")
    parser.add_argument(
        "--approaches",
        default="alphabench,quantaalpha,rd_agent,alpha_jungle,factor_lab",
        help="comma-separated approach names",
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--correlation-threshold", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_OUTPUT_DIR)
    args = parser.parse_args()

    load_dotenv()
    tickers = tuple(item.strip() for item in args.tickers.split(",") if item.strip()) or None
    approaches = tuple(item.strip() for item in args.approaches.split(",") if item.strip())
    result = run_compared_approaches(
        model=args.model,
        panel_path=args.crypto_panel,
        tickers=tickers,
        approach_names=approaches,
        count=args.count,
        top_k=args.top_k,
        correlation_threshold=args.correlation_threshold,
        output_dir=args.output_dir,
    )
    print(
        "compared approaches: "
        f"model={args.model}, approaches={len(result['results'])}, "
        f"summary={result['output_dir'] / 'summary.json'}"
    )
    for row in result["summary"]["results"]:
        print(
            f"{row['approach']}: "
            f"valid={row['valid_candidate_count']}, selected={row['selected_count']}, "
            f"test_rankic={row['test']['rank_ic_mean']:.6f}, "
            f"test_ls_sharpe={row['test']['long_short_sharpe']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
