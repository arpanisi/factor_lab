import pandas as pd
import pytest

from src.data.adapters import adapt_crypto_ohlcv, adapt_crsp_dsf_v2, adapt_taq_features


def test_adapt_crsp_dsf_v2_maps_expected_columns():
    raw = pd.DataFrame(
        {
            "dlycaldt": ["2024-01-02", "2024-01-03"],
            "dlyopen": [10, 11],
            "dlyhigh": [12, 13],
            "dlylow": [9, 10],
            "dlyclose": [11, 12],
            "dlyvol": [1000, 1100],
            "dlyret": [0.01, 0.02],
            "dlycap": [1_000_000, 1_100_000],
        }
    )
    out = adapt_crsp_dsf_v2(raw)
    assert list(out.columns) == ["dlyopen", "dlyhigh", "dlylow", "dlyclose", "dlyvol", "dlyret", "dlycap"]
    assert isinstance(out.index, pd.DatetimeIndex)


def test_adapt_crypto_ohlcv_computes_returns_when_missing():
    raw = pd.DataFrame(
        {
            "open": [10, 11],
            "high": [12, 13],
            "low": [9, 10],
            "close": [10, 12],
            "volume": [100, 120],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    out = adapt_crypto_ohlcv(raw)
    assert out.loc[pd.Timestamp("2024-01-02"), "returns"] == pytest.approx(0.2)


def test_adapt_taq_features_requires_microstructure_columns():
    raw = pd.DataFrame(
        {
            "spread": [0.01],
            "midret": [0.001],
            "imbalance": [0.2],
            "trade_size": [100],
            "trade_count": [10],
            "volume": [1000],
        },
        index=pd.to_datetime(["2024-01-01 09:30"]),
    )
    out = adapt_taq_features(raw)
    assert list(out.columns) == ["spread", "midret", "imbalance", "trade_size", "trade_count", "volume"]


def test_adapter_missing_columns_fail():
    with pytest.raises(ValueError, match="missing required columns"):
        adapt_crypto_ohlcv(pd.DataFrame({"close": [1.0]}))

