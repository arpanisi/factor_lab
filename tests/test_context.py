import numpy as np
import pandas as pd
import pytest

from src.dsl.context import PointInTimeContext


def _toy_crsp_frame():
    return pd.DataFrame(
        {
            "dlyret": [0.01, 0.02, -0.01, 0.03, 0.04],
            "dlyclose": [100.0, 102.0, 101.0, 104.0, 108.0],
            "dlyvol": [10, 20, 30, 40, 50],
        },
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        ),
    )


def _toy_crypto_frame():
    return pd.DataFrame(
        {
            "returns": [0.10, -0.05, 0.02],
            "close": [10.0, 9.5, 9.7],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )


def test_context_registers_available_namespaces():
    ctx = PointInTimeContext(
        {"crsp": _toy_crsp_frame(), "crypto": _toy_crypto_frame()},
        timestamp="2024-01-03",
    )
    assert ctx.namespaces() == ("crsp", "crypto")
    assert ctx.has_namespace("crsp")
    assert ctx.has_namespace("crypto")
    assert not ctx.has_namespace("taq")


def test_window_excludes_future_rows():
    ctx = PointInTimeContext({"crsp": _toy_crsp_frame()}, timestamp="2024-01-03")
    window = ctx.window("crsp.dlyret", 10)

    assert window.size == 3
    assert window.index.max() == pd.Timestamp("2024-01-03")
    assert pd.Timestamp("2024-01-04") not in window.index
    np.testing.assert_allclose(window.values, np.array([0.01, 0.02, -0.01]))


def test_window_uses_tail_window_up_to_timestamp():
    ctx = PointInTimeContext({"crsp": _toy_crsp_frame()}, timestamp="2024-01-04")
    window = ctx.window("crsp.dlyclose", 2)

    assert list(window.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    np.testing.assert_allclose(window.values, np.array([101.0, 104.0]))


def test_latest_returns_last_available_value():
    ctx = PointInTimeContext({"crsp": _toy_crsp_frame()}, timestamp="2024-01-04")
    assert ctx.latest("crsp.dlyvol") == pytest.approx(40.0)


def test_history_length_counts_only_visible_history():
    ctx = PointInTimeContext({"crsp": _toy_crsp_frame()}, timestamp="2024-01-02")
    assert ctx.history_length("crsp.dlyret") == 2


def test_missing_namespace_fails():
    ctx = PointInTimeContext({"crsp": _toy_crsp_frame()}, timestamp="2024-01-03")
    with pytest.raises(ValueError, match="not loaded"):
        ctx.window("crypto.returns", 2)


def test_missing_column_fails():
    ctx = PointInTimeContext({"crsp": _toy_crsp_frame()[["dlyret"]]}, timestamp="2024-01-03")
    with pytest.raises(ValueError, match="missing column"):
        ctx.window("crsp.dlyclose", 2)


def test_invalid_window_fails():
    ctx = PointInTimeContext({"crsp": _toy_crsp_frame()}, timestamp="2024-01-03")
    with pytest.raises(ValueError, match="window must be positive"):
        ctx.window("crsp.dlyret", 0)
