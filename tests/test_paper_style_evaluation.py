import pandas as pd

from src.evaluation import PostSelectionConfig, evaluate_factor_library, select_decorrelated_factors
from src.evaluation.post_selection import FactorEvaluation


def _crypto_frame(close, volume=1000.0):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    close = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": float(volume),
            "returns": close.pct_change(),
        }
    )


def test_select_decorrelated_factors_applies_threshold():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    matrices = {
        "a": pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 3, 4, 5]}, index=idx),
        "b": pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 3, 4, 5]}, index=idx),
        "c": pd.DataFrame({"x": [1, 4, 1, 4], "y": [4, 1, 4, 1]}, index=idx),
    }
    evaluations = [
        FactorEvaluation("a", "validation", 0.5, 0.3, 0.3, 1.0, 0.0, 0.0, 4, 2.0),
        FactorEvaluation("b", "validation", 0.5, 0.2, 0.2, 1.0, 0.0, 0.0, 4, 2.0),
        FactorEvaluation("c", "validation", 0.5, 0.1, 0.1, 1.0, 0.0, 0.0, 4, 2.0),
    ]

    selected = select_decorrelated_factors(evaluations, matrices, threshold=0.7, top_k=3)

    assert selected == ["a", "c"]


def test_evaluate_factor_library_returns_fused_validation_and_test_metrics():
    frames = {
        "winner": _crypto_frame([10, 11, 12, 13, 14, 15, 16, 17]),
        "middle": _crypto_frame([10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5]),
        "loser": _crypto_frame([10, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8, 8.6]),
    }

    result = evaluate_factor_library(
        ["ts_mean(crypto.returns(2))", "neg(ts_mean(crypto.returns(2)))"],
        "crypto",
        frames,
        price_col="close",
        config=PostSelectionConfig(
            validation_start="2024-01-01",
            validation_end="2024-01-05",
            test_start="2024-01-06",
            test_end="2024-01-08",
            top_k=1,
            min_history=3,
            min_assets=3,
        ),
    )

    assert len(result.selected_exprs) == 1
    assert result.validation.valid_times > 0
    assert result.test.valid_times > 0
