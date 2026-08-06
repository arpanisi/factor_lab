"""Verl-compatible executable reward function.

Verl imports this module on reward workers. The reward function receives model
completions, extracts one Factor DSL expression per completion, executes the
factor backtest, applies DiCo reward shaping, and returns a torch tensor.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from examples.build_seed_bank import crypto_frames_from_panel
from src.rft import MinedFactorDatabase
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank
from src.training.grpo_reward import FactorRewardRuntime, make_grpo_reward_func


def _archive_path() -> Path:
    return Path(os.getenv("FACTOR_LAB_ARCHIVE_JSONL", "outputs/verl/mined_factors.jsonl"))


def reward_fn(completions: Sequence[str], **metadata: Any):
    """Return one executable scalar reward per generated completion.

    Environment variables used by the worker:
    - ``FACTOR_LAB_CRYPTO_PANEL``: path to saved crypto panel pickle.
    - ``FACTOR_LAB_TICKERS``: comma-separated tickers.
    - ``FACTOR_LAB_ARCHIVE_JSONL``: mined factor database path.
    - ``FACTOR_LAB_REWARD_LOG_JSONL``: optional rollout log path.
    """

    runtime = _runtime_from_metadata(metadata)
    rewards = make_grpo_reward_func(runtime)(completions)

    # Persist any factors accepted during this call back to disk. Without this,
    # runtime.archive (loaded fresh from FACTOR_LAB_ARCHIVE_JSONL on every call)
    # is mutated only in memory and discarded, so the archive never grows across
    # training steps and the diversity/complementarity reward terms never see
    # factors discovered in earlier steps.
    archive_path = _archive_path()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.archive.save_jsonl(archive_path)

    try:
        import torch

        return torch.tensor(rewards, dtype=torch.float32)
    except Exception:
        return rewards


def _runtime_from_metadata(metadata: dict[str, Any]) -> FactorRewardRuntime:
    extra = _extra_info(metadata)
    namespace = str(extra.get("namespace") or os.getenv("FACTOR_LAB_NAMESPACE", "crypto"))
    if namespace != "crypto":
        raise ValueError("Verl reward_fn currently supports namespace='crypto' for executable GPU runs")

    panel_path = Path(os.getenv("FACTOR_LAB_CRYPTO_PANEL", "data/crypto/crypto_panel_clean.pkl"))
    tickers = tuple(
        item.strip()
        for item in os.getenv("FACTOR_LAB_TICKERS", "BTC-USD,ETH-USD,XRP-USD").split(",")
        if item.strip()
    )
    frames = _load_crypto_frames(str(panel_path), tickers)
    archive = MinedFactorDatabase.load_jsonl(_archive_path())
    task = _task_from_extra(extra)
    reward_log = os.getenv("FACTOR_LAB_REWARD_LOG_JSONL")
    return FactorRewardRuntime(
        task=task,
        namespace=namespace,
        data=frames,
        price_col="close",
        archive=archive,
        min_history=int(os.getenv("FACTOR_LAB_MIN_HISTORY", "30")),
        min_assets=max(3, min(5, len(frames))),
        reward_log_jsonl=Path(reward_log) if reward_log else None,
    )


def _extra_info(metadata: dict[str, Any]) -> dict[str, Any]:
    extra = metadata.get("extra_info") or metadata.get("extra_infos") or {}
    if isinstance(extra, list):
        return dict(extra[0] or {}) if extra else {}
    return dict(extra)


def _task_from_extra(extra: dict[str, Any]):
    scenario_payload = dict(extra.get("scenario") or {})
    window_payload = dict(extra.get("window") or {})
    scenario = FactorScenario.from_benchmark(
        str(scenario_payload.get("benchmark") or extra.get("reward_kind") or "daily_cross_sectional_rankic"),
        market=str(scenario_payload.get("market") or "crypto"),
        horizon=int(scenario_payload.get("horizon") or extra.get("horizon") or 1),
        name=str(scenario_payload.get("name") or "crypto_verl_factor_discovery"),
    )
    start = str(window_payload.get("start") or os.getenv("FACTOR_LAB_WINDOW_START", "2019-01-01"))
    end = str(window_payload.get("end") or os.getenv("FACTOR_LAB_WINDOW_END", "2026-06-16"))
    seed_expr = str(extra.get("seed_expr") or "ts_mean(crypto.returns(5))")
    seed_score = float(extra.get("seed_score") or 0.0)
    return build_task_bank(
        [SeedCandidate(seed_expr, seed_score, "")],
        scenario,
        [EvaluationWindow(start, end)],
    )[0]


@lru_cache(maxsize=4)
def _load_crypto_frames(panel_path: str, tickers: tuple[str, ...]):
    panel = pd.read_pickle(panel_path)
    return crypto_frames_from_panel(panel, tickers=tickers or None)
