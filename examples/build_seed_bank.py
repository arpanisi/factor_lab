"""Build empirically scored seed banks from local crypto data or WRDS CRSP."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import adapt_crypto_ohlcv, adapt_crsp_dsf_v2
from examples.dsl_smoke_test import (
    DEFAULT_CRYPTO_PANEL,
    fetch_wrds_crsp_dsf_v2_sample,
    load_dotenv,
)
from src.seeds import EvaluationWindow, FactorScenario, SeedPoolConfig, build_scenario_seed_bank
from src.seeds.candidates import candidate_templates_for_scenario
from src.seeds.oracle import (
    OpenRouterConfig,
    generate_oracle_seed_candidates,
    generate_oracle_seed_candidates_from_file,
)


def crypto_frames_from_panel(panel: dict, tickers: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    """Convert saved crypto panel dict into per-asset DSL-ready frames."""

    required = ["open", "high", "low", "close", "volume"]
    missing = [key for key in required if key not in panel]
    if missing:
        raise ValueError(f"crypto panel missing keys: {missing}")

    available = tuple(str(col) for col in panel["close"].columns)
    selected = tickers or available
    frames = {}
    for ticker in selected:
        frame = pd.DataFrame({key: panel[key][ticker] for key in required})
        if "returns" in panel:
            frame["returns"] = panel["returns"][ticker]
        frames[str(ticker)] = adapt_crypto_ohlcv(frame)
    return frames


def build_crypto_cross_sectional_seed_bank(
    *,
    panel_path: Path = DEFAULT_CRYPTO_PANEL,
    tickers: tuple[str, ...] | None = None,
    top_k: int = 5,
    quality_threshold: float = -1.0,
    raw_candidates: tuple[str, ...] | None = None,
) -> dict:
    """Build a cross-sectional crypto seed bank from the saved project panel."""

    panel = pd.read_pickle(panel_path)
    frames = crypto_frames_from_panel(panel, tickers=tickers)
    close = panel["close"]
    windows = [
        EvaluationWindow(str(close.index.min().date()), str(close.index.max().date())),
    ]
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="crypto_cross_sectional_seed_bank",
    )
    result = build_scenario_seed_bank(
        scenario,
        namespace="crypto",
        data=frames,
        price_col="close",
        windows=windows,
        raw_candidates=raw_candidates,
        min_history=30,
        min_assets=max(3, min(5, len(frames))),
        pool_config=SeedPoolConfig(top_k=top_k, quality_threshold=quality_threshold),
    )
    return {"result": result, "asset_count": len(frames)}


def build_wrds_crsp_single_asset_seed_bank(
    *,
    permno: int = 14593,
    start_date: str = "2023-01-01",
    end_date: str = "2023-06-30",
    top_k: int = 5,
    quality_threshold: float = 0.0,
    raw_candidates: tuple[str, ...] | None = None,
) -> dict:
    """Build a single-asset CRSP seed bank from a tiny live WRDS sample."""

    raw = fetch_wrds_crsp_dsf_v2_sample(permno=permno, start_date=start_date, end_date=end_date)
    frame = adapt_crsp_dsf_v2(raw, date_col="dlycaldt")
    windows = [EvaluationWindow(start_date, end_date)]
    scenario = FactorScenario.from_benchmark(
        "single_asset_direction",
        market="wrds_crsp",
        horizon=1,
        name="wrds_crsp_single_asset_seed_bank",
    )
    result = build_scenario_seed_bank(
        scenario,
        namespace="crsp",
        data=frame,
        price_col="dlyclose",
        windows=windows,
        raw_candidates=raw_candidates,
        min_history=20,
        pool_config=SeedPoolConfig(top_k=top_k, quality_threshold=quality_threshold),
    )
    return {"result": result, "row_count": int(frame.shape[0])}


def print_seed_bank_summary(label: str, payload: dict) -> None:
    """Print a compact seed-bank summary."""

    result = payload["result"]
    print(f"{label}: raw={result.raw_candidate_count}, scored={result.scored_candidate_count}, seeds={len(result.seeds)}, tasks={len(result.tasks)}")
    for i, seed in enumerate(result.seeds, start=1):
        print(f"{i}. score={seed.score:.6f} expr={seed.expr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Factor Lab seed banks.")
    parser.add_argument("--crypto-panel", type=Path, default=DEFAULT_CRYPTO_PANEL)
    parser.add_argument("--tickers", default="", help="comma-separated crypto tickers; default uses all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quality-threshold", type=float, default=-1.0)
    parser.add_argument("--candidate-source", choices=("template", "openrouter", "file"), default="template")
    parser.add_argument("--oracle-output-file", type=Path)
    parser.add_argument("--oracle-count", type=int, default=24)
    parser.add_argument("--openrouter-model", default="deepseek/deepseek-chat")
    parser.add_argument("--wrds-crsp", action="store_true", help="also build a tiny live WRDS CRSP seed bank")
    parser.add_argument("--wrds-permno", type=int, default=14593)
    parser.add_argument("--wrds-start", default="2023-01-01")
    parser.add_argument("--wrds-end", default="2023-06-30")
    args = parser.parse_args()

    load_dotenv()
    tickers = tuple(item.strip() for item in args.tickers.split(",") if item.strip()) or None
    crypto_scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="crypto_cross_sectional_seed_bank",
    )
    raw_candidates = _load_raw_candidates(
        crypto_scenario,
        namespace="crypto",
        source=args.candidate_source,
        output_file=args.oracle_output_file,
        count=args.oracle_count,
        model=args.openrouter_model,
    )
    crypto = build_crypto_cross_sectional_seed_bank(
        panel_path=args.crypto_panel,
        tickers=tickers,
        top_k=args.top_k,
        quality_threshold=args.quality_threshold,
        raw_candidates=raw_candidates,
    )
    print_seed_bank_summary("crypto cross-sectional seed bank", crypto)

    if args.wrds_crsp:
        try:
            crsp_scenario = FactorScenario.from_benchmark(
                "single_asset_direction",
                market="wrds_crsp",
                horizon=1,
                name="wrds_crsp_single_asset_seed_bank",
            )
            crsp_raw_candidates = _load_raw_candidates(
                crsp_scenario,
                namespace="crsp",
                source=args.candidate_source,
                output_file=args.oracle_output_file,
                count=args.oracle_count,
                model=args.openrouter_model,
            )
            wrds_payload = build_wrds_crsp_single_asset_seed_bank(
                permno=args.wrds_permno,
                start_date=args.wrds_start,
                end_date=args.wrds_end,
                top_k=args.top_k,
                quality_threshold=args.quality_threshold,
                raw_candidates=crsp_raw_candidates,
            )
        except RuntimeError as exc:
            print(f"wrds crsp seed bank failed: {exc}")
            return 1
        print_seed_bank_summary("wrds crsp single-asset seed bank", wrds_payload)

    return 0


def _load_raw_candidates(
    scenario: FactorScenario,
    *,
    namespace: str,
    source: str,
    output_file: Path | None,
    count: int,
    model: str,
) -> tuple[str, ...] | None:
    if source == "template":
        return candidate_templates_for_scenario(scenario)
    if source == "file":
        if output_file is None:
            raise ValueError("--oracle-output-file is required when --candidate-source=file")
        return generate_oracle_seed_candidates_from_file(output_file, namespace=namespace)
    if source == "openrouter":
        return generate_oracle_seed_candidates(
            scenario,
            namespace=namespace,
            count=count,
            config=OpenRouterConfig(model=model),
        )
    raise ValueError(f"unknown candidate source: {source}")


if __name__ == "__main__":
    raise SystemExit(main())
