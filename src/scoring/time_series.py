"""Single-asset empirical scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.scoring.series import evaluate_factor_series, forward_returns


@dataclass(frozen=True)
class TimeSeriesScore:
    """Directional and IC metrics for one factor on one asset."""

    score: float
    directional_accuracy: float
    ic: float
    rank_ic: float
    samples: int


def score_directional_prediction(
    expr: str,
    namespace: str,
    frame: pd.DataFrame,
    *,
    price_col: str,
    horizon: int = 1,
    min_history: int = 1,
) -> TimeSeriesScore:
    """Score whether a factor predicts next-period return direction."""

    factor = evaluate_factor_series(expr, namespace, frame, min_history=min_history)
    fwd = forward_returns(frame, price_col=price_col, horizon=horizon)
    aligned = pd.concat({"factor": factor, "fwd": fwd}, axis=1).dropna()

    if aligned.empty:
        return TimeSeriesScore(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    signed_factor = np.sign(aligned["factor"].to_numpy(dtype=float))
    signed_fwd = np.sign(aligned["fwd"].to_numpy(dtype=float))
    active = signed_factor != 0
    if active.sum() == 0:
        dir_acc = float("nan")
    else:
        dir_acc = float(np.mean(signed_factor[active] == signed_fwd[active]))

    ic = _corr(aligned["factor"], aligned["fwd"], method="pearson")
    rank_ic = _corr(aligned["factor"], aligned["fwd"], method="spearman")
    score = dir_acc if np.isfinite(dir_acc) else -1.0
    return TimeSeriesScore(score, dir_acc, ic, rank_ic, int(aligned.shape[0]))


def _corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    if left.shape[0] < 2:
        return float("nan")
    if method == "spearman":
        left = left.rank(method="average")
        right = right.rank(method="average")
    if float(left.std(ddof=0)) <= 1e-12 or float(right.std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(left.corr(right))
