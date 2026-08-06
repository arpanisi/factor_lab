import numpy as np
import pandas as pd
import pytest

from src.scoring import (
    evaluate_factor_series,
    forward_returns,
    score_cross_sectional_rankic,
    score_directional_prediction,
)
from src.seeds import SeedPoolConfig, build_seed_pool


def _crypto_frame(close):
    close = pd.Series(close, index=pd.date_range("2024-01-01", periods=len(close), freq="D"), dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
            "returns": close.pct_change(),
        }
    )


def test_evaluate_factor_series_uses_point_in_time_history():
    frame = _crypto_frame([10, 11, 13, 16])

    out = evaluate_factor_series("last(crypto.close(2))", "crypto", frame, min_history=2)

    assert list(out.index) == list(frame.index[1:])
    assert out.iloc[0] == pytest.approx(11.0)
    assert out.iloc[-1] == pytest.approx(16.0)


def test_forward_returns():
    frame = _crypto_frame([10, 12, 15])

    out = forward_returns(frame, price_col="close", horizon=1)

    assert out.iloc[0] == pytest.approx(0.2)
    assert np.isnan(out.iloc[-1])


def test_directional_prediction_scores_known_signal():
    frame = _crypto_frame([10, 11, 12, 11, 10, 12])

    result = score_directional_prediction(
        "sign(diff(crypto.close(2)))",
        "crypto",
        frame,
        price_col="close",
        min_history=2,
    )

    assert result.samples > 0
    assert 0.0 <= result.directional_accuracy <= 1.0
    assert result.score == result.directional_accuracy


def test_cross_sectional_rankic_scores_ranked_assets():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    frames = {
        "winner": _crypto_frame([10, 11, 12, 13, 14, 15]).loc[dates],
        "middle": _crypto_frame([10, 10.5, 11, 11.5, 12, 12.5]).loc[dates],
        "loser": _crypto_frame([10, 9.8, 9.6, 9.4, 9.2, 9.0]).loc[dates],
    }

    result = score_cross_sectional_rankic(
        "ts_mean(crypto.returns(2))",
        "crypto",
        frames,
        price_col="close",
        min_history=3,
        min_assets=3,
    )

    assert result.valid_times > 0
    assert result.rank_ic_mean > 0.0
    assert result.score > 0.0


def test_seed_pool_can_use_empirical_score_callback():
    frame = _crypto_frame([10, 11, 12, 11, 10, 12, 13])

    def score_fn(expr):
        return score_directional_prediction(
            expr,
            "crypto",
            frame,
            price_col="close",
            min_history=2,
        ).score

    seeds = build_seed_pool(
        ["sign(diff(crypto.close(2)))", "future_return(crypto.close(2))"],
        score_fn=score_fn,
        config=SeedPoolConfig(top_k=3, quality_threshold=0.0),
    )

    assert len(seeds) == 1
    assert seeds[0].expr == "sign(diff(crypto.close(2)))"
