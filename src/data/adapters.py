"""Adapters from raw datasets into Factor Lab namespace schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
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


def verify_crypto_panel(
    panel_or_path: str | Path | dict,
    tickers: Iterable[str] | None = None,
) -> dict:
    """Verify crypto panel existence and schema non-degeneracy per §9.1.

    Raises FileNotFoundError naming the missing path if a path was passed and doesn't exist.
    Raises ValueError naming any missing required keys or any missing tickers.
    """

    if isinstance(panel_or_path, (str, Path)):
        path = Path(panel_or_path)
        if not path.exists():
            raise FileNotFoundError(f"crypto panel file not found: '{path}'")
        panel = pd.read_pickle(path)
    elif isinstance(panel_or_path, dict):
        panel = panel_or_path
    else:
        raise ValueError(f"expected dict or path for crypto panel, got {type(panel_or_path).__name__}")

    required = ["open", "high", "low", "close", "volume"]
    missing = [key for key in required if key not in panel]
    if missing:
        raise ValueError(f"crypto panel missing required key(s): {missing}")

    if tickers is not None:
        ticker_list = tuple(str(t) for t in tickers)
        if hasattr(panel["close"], "columns"):
            available = set(panel["close"].columns)
        elif isinstance(panel["close"], dict):
            available = set(panel["close"].keys())
        else:
            available = set()
        missing_tickers = [t for t in ticker_list if t not in available]
        if missing_tickers:
            raise ValueError(f"crypto panel missing requested ticker(s): {missing_tickers}")

    return panel


