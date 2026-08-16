import pandas as pd
import pytest

from src.data.adapters import (
    adapt_crsp_dsf_v2,
    adapt_crypto_ohlcv,
    adapt_taq_features,
    verify_crypto_panel,
)


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


def test_verify_crypto_panel_missing_file_raises_file_not_found(tmp_path):
    missing = tmp_path / "nonexistent.pkl"
    with pytest.raises(FileNotFoundError, match="crypto panel file not found"):
        verify_crypto_panel(missing)


def test_verify_crypto_panel_missing_keys_raises_value_error():
    incomplete = {
        "open": pd.DataFrame({"BTC": [10.0]}),
        "close": pd.DataFrame({"BTC": [10.0]}),
    }
    with pytest.raises(ValueError, match="crypto panel missing required key"):
        verify_crypto_panel(incomplete)


def test_verify_crypto_panel_missing_tickers_raises_value_error():
    panel = {
        "open": pd.DataFrame({"BTC": [10.0]}),
        "high": pd.DataFrame({"BTC": [11.0]}),
        "low": pd.DataFrame({"BTC": [9.0]}),
        "close": pd.DataFrame({"BTC": [10.0]}),
        "volume": pd.DataFrame({"BTC": [100.0]}),
    }
    with pytest.raises(ValueError, match=r"crypto panel missing requested ticker\(s\): \['ETH'\]"):
        verify_crypto_panel(panel, tickers=["BTC", "ETH"])


def test_verify_crypto_panel_valid_panel(tmp_path):
    panel = {
        "open": pd.DataFrame({"BTC": [10.0], "ETH": [20.0]}),
        "high": pd.DataFrame({"BTC": [11.0], "ETH": [21.0]}),
        "low": pd.DataFrame({"BTC": [9.0], "ETH": [19.0]}),
        "close": pd.DataFrame({"BTC": [10.0], "ETH": [20.0]}),
        "volume": pd.DataFrame({"BTC": [100.0], "ETH": [200.0]}),
    }
    path = tmp_path / "panel.pkl"
    pd.to_pickle(panel, path)

    loaded = verify_crypto_panel(path, tickers=["BTC", "ETH"])
    assert set(loaded.keys()) >= {"open", "high", "low", "close", "volume"}
    assert set(loaded["close"].columns) == {"BTC", "ETH"}


