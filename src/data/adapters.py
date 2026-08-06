"""Adapters from raw datasets into Factor Lab namespace schemas."""

from __future__ import annotations

import pandas as pd


def _copy_with_datetime_index(frame: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    if date_col is not None:
        if date_col not in out.columns:
            raise ValueError(f"missing date column '{date_col}'")
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.set_index(date_col)
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    out = out.sort_index()
    return out


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def adapt_crsp_dsf_v2(frame: pd.DataFrame, date_col: str | None = "dlycaldt") -> pd.DataFrame:
    """Adapt WRDS CRSP ``dsf_v2`` rows for one asset into DSL-ready columns.

    Output columns are WRDS-native DSL fields:
    ``dlyopen, dlyhigh, dlylow, dlyclose, dlyvol, dlyret, dlycap``.
    """

    required = ["dlyopen", "dlyhigh", "dlylow", "dlyclose", "dlyvol", "dlyret", "dlycap"]
    out = _copy_with_datetime_index(frame, date_col=date_col)
    _require_columns(out, required)
    return out[required].apply(pd.to_numeric, errors="coerce")


def adapt_crypto_ohlcv(frame: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """Adapt crypto OHLCV rows into DSL-ready columns.

    If ``returns`` is missing, it is computed from close-to-close returns.
    """

    required = ["open", "high", "low", "close", "volume"]
    out = _copy_with_datetime_index(frame, date_col=date_col)
    _require_columns(out, required)
    out = out[required + (["returns"] if "returns" in out.columns else [])].apply(
        pd.to_numeric, errors="coerce"
    )
    if "returns" not in out.columns:
        out["returns"] = out["close"].pct_change()
    return out[["open", "high", "low", "close", "volume", "returns"]]


def adapt_taq_features(frame: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """Adapt precomputed TAQ features into DSL-ready columns."""

    required = ["spread", "midret", "imbalance", "trade_size", "trade_count", "volume"]
    out = _copy_with_datetime_index(frame, date_col=date_col)
    _require_columns(out, required)
    return out[required].apply(pd.to_numeric, errors="coerce")

