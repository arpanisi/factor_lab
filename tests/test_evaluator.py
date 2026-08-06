import numpy as np
import pandas as pd

from src.dsl.context import PointInTimeContext
from src.dsl.evaluator import evaluate_expr


def test_evaluate_crsp_scalar_expression_point_in_time():
    df = pd.DataFrame(
        {
            "dlyret": [0.01, 0.02, 0.03, 0.04],
            "dlyclose": [100, 102, 105, 109],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    ctx = PointInTimeContext({"crsp": df}, "2024-01-03")
    value = evaluate_expr("ts_mean(crsp.dlyret(2))", ctx)
    assert value == np.mean([0.02, 0.03])


def test_evaluate_expression_does_not_use_future_data():
    df = pd.DataFrame(
        {"dlyret": [0.01, 0.02, 100.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    ctx = PointInTimeContext({"crsp": df}, "2024-01-02")
    value = evaluate_expr("ts_max(crsp.dlyret(10))", ctx)
    assert value == 0.02


def test_evaluate_crypto_expression():
    df = pd.DataFrame(
        {"returns": [0.01, 0.03, 0.05]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    ctx = PointInTimeContext({"crypto": df}, "2024-01-03")
    value = evaluate_expr("div(ts_mean(crypto.returns(3)), ts_std(crypto.returns(3)))", ctx)
    assert np.isfinite(value)


def test_array_expression_returns_last_value():
    df = pd.DataFrame(
        {"dlyret": [0.01, 0.02, 0.03]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    ctx = PointInTimeContext({"crsp": df}, "2024-01-03")
    value = evaluate_expr("zscore(crsp.dlyret(3))", ctx)
    assert value == np.array([-1.224744871391589, 0.0, 1.224744871391589])[-1]
