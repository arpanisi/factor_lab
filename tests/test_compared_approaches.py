import pandas as pd

from src.benchmarks import run_compared_approaches


def test_run_compared_approaches_with_fake_generator(tmp_path):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    tickers = ["ADA-USD", "BNB-USD", "BTC-USD", "DOGE-USD", "ETH-USD", "LINK-USD", "XLM-USD", "XRP-USD"]
    panel = {
        "open": pd.DataFrame(
            {t: range(100 + i * 10, 140 + i * 10) for i, t in enumerate(tickers)},
            index=dates,
            dtype=float,
        )
    }
    panel["high"] = panel["open"] + 1
    panel["low"] = panel["open"] - 1
    panel["close"] = panel["open"]
    panel["volume"] = pd.DataFrame(
        {t: range(1000 + i * 50, 1040 + i * 50) for i, t in enumerate(tickers)},
        index=dates,
        dtype=float,
    )
    panel["returns"] = panel["close"].pct_change()
    panel_path = tmp_path / "panel.pkl"
    pd.to_pickle(panel, panel_path)

    def fake_generator(scenario, approach, frames, price_col, model, count):
        return (
            "ts_mean(crypto.returns(2))",
            "neg(ts_mean(crypto.returns(2)))",
            "div(ts_mean(crypto.volume(5)), ts_std(crypto.returns(5)))",
        )

    result = run_compared_approaches(
        model="fake/model",
        panel_path=panel_path,
        approach_names=("alphabench", "quantaalpha"),
        count=3,
        top_k=2,
        output_dir=tmp_path / "out",
        generator=fake_generator,
    )

    assert len(result["results"]) == 2
    assert (tmp_path / "out" / "summary.json").exists()
    assert result["summary"]["results"][0]["selected_count"] > 0
