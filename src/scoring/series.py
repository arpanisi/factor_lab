"""Shared factor-series evaluation helpers."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.dsl import PointInTimeContext, evaluate_expr


def evaluate_factor_series(
    expr: str,
    namespace: str,
    frame: pd.DataFrame,
    *,
    min_history: int = 1,
) -> pd.Series:
    """Evaluate a DSL expression at each timestamp of one asset frame."""

    if frame.empty:
        return pd.Series(dtype=float)

    values: list[float] = []
    index = []
    clean = frame.sort_index()

    for i, timestamp in enumerate(clean.index):
        if i + 1 < int(min_history):
            continue
        ctx = PointInTimeContext({namespace: clean.iloc[: i + 1]}, timestamp)
        try:
            value = evaluate_expr(expr, ctx)
        except Exception:
            value = float("nan")
        values.append(float(value))
        index.append(timestamp)

    return pd.Series(values, index=pd.DatetimeIndex(index), dtype=float).replace([float("inf"), -float("inf")], np.nan)


def forward_returns(frame: pd.DataFrame, *, price_col: str, horizon: int = 1) -> pd.Series:
    """Compute forward simple returns from a price column."""

    close = pd.to_numeric(frame[price_col], errors="coerce")
    return close.shift(-int(horizon)) / close - 1.0
