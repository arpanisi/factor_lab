import pandas as pd

from src.benchmarks import run_compared_approaches


def test_run_compared_approaches_with_fake_generator(tmp_path):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    panel = {
        "open": pd.DataFrame(
            {
                "BTC-USD": range(100, 140),
                "ETH-USD": range(80, 120),
                "XRP-USD": range(40, 80),
            },
            index=dates,
            dtype=float,
        )
    }
    panel["high"] = panel["open"] + 1
    panel["low"] = panel["open"] - 1
    panel["close"] = panel["open"]
    panel["volume"] = pd.DataFrame(
        {
            "BTC-USD": range(1000, 1040),
            "ETH-USD": range(900, 940),
            "XRP-USD": range(800, 840),
        },
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
