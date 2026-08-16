"""Run baseline oracle/miner rollouts with OpenRouter models."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from examples.build_seed_bank import crypto_frames_from_panel
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL, load_dotenv
from src.data import verify_crypto_panel
from src.rft import (
    DatabaseSelectionConfig,
    MinerConfig,
    MinedFactorDatabase,
    run_discovery_for_task,
)
from src.seeds import (
    EvaluationWindow,
    FactorScenario,
    SeedPoolConfig,
    build_scenario_seed_bank,
    generate_oracle_seed_candidates,
)
from src.seeds.oracle import OpenRouterConfig


DEFAULT_OUTPUT_DIR = Path("factor_lab") / "outputs" / "baseline_rollouts"


def run_crypto_baseline_rollout(
    *,
    oracle_model: str = "deepseek/deepseek-chat-v3.1",
    miner_model: str = "qwen/qwen3-235b-a22b-2507",
    panel_path: Path = DEFAULT_CRYPTO_PANEL,
    tickers: tuple[str, ...] | None = None,
    oracle_count: int = 24,
    miner_count: int = 4,
    top_k_seeds: int = 3,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run oracle seed generation plus miner baseline rollouts on crypto data."""

    load_dotenv()
    panel = verify_crypto_panel(panel_path, tickers=tickers)
    frames = crypto_frames_from_panel(panel, tickers=tickers)
    close = panel["close"]
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="crypto_cross_sectional_baseline_rollout",
    )
    windows = [EvaluationWindow(str(close.index.min().date()), str(close.index.max().date()))]

    raw_candidates = generate_oracle_seed_candidates(
        scenario,
        namespace="crypto",
        count=oracle_count,
        config=OpenRouterConfig(model=oracle_model),
    )
    seed_build = build_scenario_seed_bank(
        scenario,
        namespace="crypto",
        data=frames,
        price_col="close",
        windows=windows,
        raw_candidates=raw_candidates,
        min_history=30,
        min_assets=8,
        pool_config=SeedPoolConfig(top_k=top_k_seeds, quality_threshold=-1.0),
    )

    archive = MinedFactorDatabase()
    rollout_records = []
    for task_id, task in enumerate(seed_build.tasks):
        results = run_discovery_for_task(
            task,
            namespace="crypto",
            data=frames,
            price_col="close",
            count=miner_count,
            min_history=30,
            min_assets=8,
            miner_config=MinerConfig(model=miner_model),
            archive=archive,
            selection_config=DatabaseSelectionConfig(min_score=-1.0, min_reward=-1.0),
        )
        for result in results:
            rollout_records.append(
                {
                    "task_id": task_id,
                    "seed_expr": task.seed_expr,
                    "seed_score": task.seed_score,
                    "expr": result.expr,
                    "valid": result.valid,
                    "score": result.score,
                    "reward": result.reward,
                    "stored": result.stored,
                    "reason": result.reason,
                    "metrics": result.metrics,
                    "reward_components": result.reward_components,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "oracle_model": oracle_model,
        "miner_model": miner_model,
        "asset_count": len(frames),
        "raw_oracle_candidates": list(raw_candidates),
        "seed_count": len(seed_build.seeds),
        "task_count": len(seed_build.tasks),
        "rollout_count": len(rollout_records),
        "stored_count": len(archive.records),
        "seeds": [asdict(seed) for seed in seed_build.seeds],
        "rollouts": rollout_records,
    }
    json_path = output_dir / "baseline_rollout.json"
    csv_path = output_dir / "baseline_rollout.csv"
    db_path = output_dir / "mined_factors.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_rollout_csv(csv_path, rollout_records)
    archive.save_jsonl(db_path)

    return {"payload": payload, "json_path": json_path, "csv_path": csv_path, "db_path": db_path}


def _write_rollout_csv(path: Path, rows: list[dict]) -> None:
    fields = ["task_id", "seed_expr", "seed_score", "expr", "valid", "score", "reward", "stored", "reason"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline oracle/miner rollouts.")
    parser.add_argument("--oracle-model", default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--miner-model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--crypto-panel", type=Path, default=DEFAULT_CRYPTO_PANEL)
    parser.add_argument("--tickers", default="ADA-USD,BNB-USD,BTC-USD,DOGE-USD,ETH-USD,LINK-USD,XLM-USD,XRP-USD")
    parser.add_argument("--oracle-count", type=int, default=24)
    parser.add_argument("--miner-count", type=int, default=4)
    parser.add_argument("--top-k-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    tickers = tuple(item.strip() for item in args.tickers.split(",") if item.strip()) or None
    result = run_crypto_baseline_rollout(
        oracle_model=args.oracle_model,
        miner_model=args.miner_model,
        panel_path=args.crypto_panel,
        tickers=tickers,
        oracle_count=args.oracle_count,
        miner_count=args.miner_count,
        top_k_seeds=args.top_k_seeds,
        output_dir=args.output_dir,
    )
    payload = result["payload"]
    print(
        "baseline rollout: "
        f"oracle={payload['oracle_model']}, miner={payload['miner_model']}, "
        f"seeds={payload['seed_count']}, tasks={payload['task_count']}, "
        f"rollouts={payload['rollout_count']}, stored={payload['stored_count']}"
    )
    print(f"json={result['json_path']}")
    print(f"csv={result['csv_path']}")
    print(f"db={result['db_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
