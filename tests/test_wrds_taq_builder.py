import pandas as pd

from src.data.wrds_taq import (
    _parse_taq_timestamp,
    _append_parquet_frame,
    build_taq_1m_features,
    build_taq_symbol_check_query,
    build_taq_wct_query,
    safe_symbol,
    taq_wct_table,
)


def test_taq_wct_table_builds_partitioned_table_name():
    assert taq_wct_table("2024-01-02") == "taqmsec.wct_20240102"


def test_build_taq_wct_query_uses_symbol_array_and_time_bounds():
    query, params = build_taq_wct_query(date="2024-01-02", symbols=["AAPL", "MSFT"])

    assert "from taqmsec.wct_20240102" in query
    assert "sym_root = any(%(symbols)s)" in query
    assert params["symbols"] == ["AAPL", "MSFT"]
    assert params["start_time"] == "09:30:00"
    assert params["end_time"] == "16:00:00"


def test_build_taq_symbol_check_query_counts_candidates():
    query, params = build_taq_symbol_check_query(date="2024-01-02", symbols=["GOOG", "GOOGL"])

    assert "from taqmsec.wct_20240102" in query
    assert "count(*) as row_count" in query
    assert params["symbols"] == ["GOOG", "GOOGL"]


def test_build_taq_1m_features_aggregates_ohlcv_and_microstructure():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "time_m": ["09:30:01", "09:30:20", "09:31:05", "09:31:40"],
            "sym_root": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "sym_suffix": ["", "", "", ""],
            "size": [100, 200, 50, 150],
            "price": [100.0, 100.2, 100.3, 100.1],
            "nbo": [100.1, 100.3, 100.4, 100.2],
            "nbb": [99.9, 100.1, 100.2, 100.0],
        }
    )

    frames = build_taq_1m_features(raw)
    out = frames["AAPL"]

    assert list(out.columns) == ["open", "high", "low", "close", "volume", "spread", "midret", "imbalance", "trade_size", "trade_count"]
    assert out.iloc[0]["open"] == 100.0
    assert out.iloc[0]["high"] == 100.2
    assert out.iloc[0]["volume"] == 300
    assert out.iloc[0]["trade_count"] == 2
    assert out.iloc[1]["close"] == 100.1


def test_safe_symbol_for_taq_file_names():
    assert safe_symbol("BRK.B") == "BRK_B"


def test_parse_taq_timestamp_handles_mixed_fractional_seconds():
    out = _parse_taq_timestamp(
        pd.Series(pd.to_datetime(["2024-01-02", "2024-01-02"])),
        pd.Series(["09:57:54.123", "09:57:54"]),
    )

    assert out.notna().all()
    assert out.iloc[0].second == 54
    assert out.iloc[1].second == 54


def test_append_parquet_frame_merges_existing_rows(tmp_path):
    path = tmp_path / "AAPL_1m.parquet"
    idx = pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:31"])
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
            "spread": [0.01, 0.01],
            "midret": [0.0, 0.1],
            "imbalance": [0.0, 0.0],
            "trade_size": [10.0, 20.0],
            "trade_count": [1.0, 1.0],
        },
        index=idx,
    )

    _append_parquet_frame(path, frame.iloc[:1])
    _append_parquet_frame(path, frame.iloc[1:])
    out = pd.read_parquet(path)

    assert len(out) == 2
    assert list(out.columns) == list(frame.columns)
