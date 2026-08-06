import numpy as np
import pytest

from src.dsl import operators as ops


def test_time_series_reductions_ignore_nan():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert ops.ts_mean(x) == pytest.approx(7.0 / 3.0)
    assert ops.ts_sum(x) == pytest.approx(7.0)
    assert ops.ts_min(x) == pytest.approx(1.0)
    assert ops.ts_max(x) == pytest.approx(4.0)


def test_first_and_last():
    assert ops.first([2.0, 3.0, 4.0]) == pytest.approx(2.0)
    assert ops.last([2.0, 3.0, 4.0]) == pytest.approx(4.0)


def test_diff_returns_array_difference():
    np.testing.assert_allclose(ops.diff([1.0, 3.0, 6.0]), np.array([2.0, 3.0]))


def test_normalize_scales_by_absolute_sum():
    np.testing.assert_allclose(ops.normalize([1.0, -3.0]), np.array([0.25, -0.75]))


def test_ema_returns_array_with_latest_smoothed_value():
    out = ops.ema(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (3,)
    assert out[-1] == pytest.approx(2.25)


def test_log_masks_non_positive_values():
    out = ops.log(np.array([1.0, np.e, 0.0, -1.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert np.isnan(out[2])
    assert np.isnan(out[3])


def test_arithmetic_operators():
    np.testing.assert_allclose(ops.add([1, 2], [3, 4]), np.array([4.0, 6.0]))
    np.testing.assert_allclose(ops.sub([1, 2], [3, 4]), np.array([-2.0, -2.0]))
    np.testing.assert_allclose(ops.mul([1, 2], [3, 4]), np.array([3.0, 8.0]))
    np.testing.assert_allclose(ops.div([2, 4], [1, 2]), np.array([2.0, 2.0]))


def test_division_by_zero_returns_nan():
    out = ops.div(np.array([1.0, 2.0]), np.array([0.0, 2.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)


def test_unary_scalar_and_array_operators():
    assert ops.neg(2.0) == pytest.approx(-2.0)
    assert ops.abs_val(-2.0) == pytest.approx(2.0)
    assert ops.sign(-2.0) == pytest.approx(-1.0)
    np.testing.assert_allclose(ops.tanh([0.0]), np.array([0.0]))


def test_corr_and_cov():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    assert ops.corr(x, y) == pytest.approx(1.0)
    assert ops.cov(x, y) == pytest.approx(np.cov(x, y, ddof=0)[0, 1])


def test_corr_constant_series_returns_nan():
    assert np.isnan(ops.corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_zscore():
    out = ops.zscore(np.array([1.0, 2.0, 3.0]))
    assert out.mean() == pytest.approx(0.0)
    assert out.std() == pytest.approx(1.0)


def test_rank_ordinal_ascending_with_nan():
    out = ops.rank(np.array([10.0, np.nan, 5.0, 7.0]))
    np.testing.assert_allclose(out, np.array([3.0, np.nan, 1.0, 2.0]), equal_nan=True)


def test_get_operator_validation():
    assert ops.get_operator("ts_mean") is ops.ts_mean
    with pytest.raises(ValueError, match="unknown operator"):
        ops.get_operator("future_return")
