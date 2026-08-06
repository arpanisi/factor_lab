"""Cross-sectional RankIC empirical scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.scoring.series import evaluate_factor_series, forward_returns


@dataclass(frozen=True)
class CrossSectionalScore:
    """Cross-sectional factor ranking metrics."""

    score: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float
    long_short_mean: float
    long_short_sharpe: float
    valid_times: int
    mean_assets_per_time: float


def score_cross_sectional_rankic(
    expr: str,
    namespace: str,
    frames_by_asset: Mapping[str, pd.DataFrame],
    *,
    price_col: str,
    horizon: int = 1,
    min_history: int = 1,
    min_assets: int = 3,
) -> CrossSectionalScore:
    """Score a factor by average cross-sectional RankIC over time."""

    factor_by_asset = {}
    fwd_by_asset = {}
    for asset, frame in frames_by_asset.items():
        factor_by_asset[str(asset)] = evaluate_factor_series(expr, namespace, frame, min_history=min_history)
        fwd_by_asset[str(asset)] = forward_returns(frame, price_col=price_col, horizon=horizon)

    factor_df = pd.concat(factor_by_asset, axis=1)
    fwd_df = pd.concat(fwd_by_asset, axis=1)
    common_index = factor_df.index.intersection(fwd_df.index)
    common_cols = factor_df.columns.intersection(fwd_df.columns)
    factor_df = factor_df.loc[common_index, common_cols]
    fwd_df = fwd_df.loc[common_index, common_cols]

    rank_ics: list[float] = []
    long_short: list[float] = []
    assets_per_time: list[int] = []

    for timestamp in common_index:
        pair = pd.concat({"factor": factor_df.loc[timestamp], "fwd": fwd_df.loc[timestamp]}, axis=1).dropna()
        if pair.shape[0] < int(min_assets):
            continue

        rank_ic = _spearman_corr(pair["factor"], pair["fwd"])
        if np.isfinite(rank_ic):
            rank_ics.append(float(rank_ic))

        spread = _long_short_spread(pair["factor"], pair["fwd"])
        if np.isfinite(spread):
            long_short.append(float(spread))
        assets_per_time.append(int(pair.shape[0]))

    if not rank_ics:
        return CrossSectionalScore(-1.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, 0.0)

    rank_arr = np.asarray(rank_ics, dtype=float)
    ls_arr = np.asarray(long_short, dtype=float)
    rank_ic_mean = float(np.mean(rank_arr))
    rank_ic_std = float(np.std(rank_arr))
    rank_ic_ir = float(rank_ic_mean / (rank_ic_std + 1e-12) * np.sqrt(rank_arr.size))
    long_short_mean = float(np.mean(ls_arr)) if ls_arr.size else float("nan")
    long_short_sharpe = (
        float(long_short_mean / (np.std(ls_arr) + 1e-12) * np.sqrt(ls_arr.size)) if ls_arr.size else float("nan")
    )
    score = float(np.clip(rank_ic_mean * 5.0 + 0.02 * np.tanh(rank_ic_ir), -1.0, 1.0))

    return CrossSectionalScore(
        score=score,
        rank_ic_mean=rank_ic_mean,
        rank_ic_std=rank_ic_std,
        rank_ic_ir=rank_ic_ir,
        long_short_mean=long_short_mean,
        long_short_sharpe=long_short_sharpe,
        valid_times=int(rank_arr.size),
        mean_assets_per_time=float(np.mean(assets_per_time)),
    )


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    if float(left_rank.std(ddof=0)) <= 1e-12 or float(right_rank.std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(left_rank.corr(right_rank))


def _long_short_spread(factor: pd.Series, fwd: pd.Series) -> float:
    n = max(1, int(np.floor(factor.shape[0] * 0.2)))
    ordered = factor.sort_values()
    short_assets = ordered.index[:n]
    long_assets = ordered.index[-n:]
    return float(fwd.loc[long_assets].mean() - fwd.loc[short_assets].mean())
