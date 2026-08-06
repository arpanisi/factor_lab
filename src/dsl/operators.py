"""Safe numerical operators for the Factor Lab DSL."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def ts_mean(x) -> float:
    values = _finite(_arr(x))
    return float(np.nan) if values.size == 0 else float(np.mean(values))


def ts_std(x) -> float:
    values = _finite(_arr(x))
    return float(np.nan) if values.size == 0 else float(np.std(values))


def ts_sum(x) -> float:
    values = _finite(_arr(x))
    return float(np.nan) if values.size == 0 else float(np.sum(values))


def ts_min(x) -> float:
    values = _finite(_arr(x))
    return float(np.nan) if values.size == 0 else float(np.min(values))


def ts_max(x) -> float:
    values = _finite(_arr(x))
    return float(np.nan) if values.size == 0 else float(np.max(values))


def first(x) -> float:
    values = _arr(x)
    return float(np.nan) if values.size == 0 else float(values[0])


def last(x) -> float:
    values = _arr(x)
    return float(np.nan) if values.size == 0 else float(values[-1])


def diff(x) -> np.ndarray:
    values = _arr(x)
    return np.diff(values)


def normalize(x) -> np.ndarray:
    values = _arr(x)
    total = np.nansum(np.abs(values))
    if not np.isfinite(total) or total <= EPS:
        return np.full_like(values, np.nan, dtype=float)
    return values / total


def ema(x) -> np.ndarray:
    values = _arr(x)
    if values.size == 0:
        return values

    alpha = 2.0 / (values.size + 1.0)
    out = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out

    start = int(np.where(finite)[0][0])
    out[start] = values[start]
    for i in range(start + 1, values.size):
        prev = out[i - 1]
        out[i] = prev if not np.isfinite(values[i]) else alpha * values[i] + (1.0 - alpha) * prev
    return out


def log(x):
    values = _arr(x)
    out = np.full_like(values, np.nan, dtype=float)
    mask = values > 0
    out[mask] = np.log(values[mask])
    return float(out) if out.ndim == 0 else out


def add(a, b):
    return _arr(a) + _arr(b)


def sub(a, b):
    return _arr(a) - _arr(b)


def mul(a, b):
    return _arr(a) * _arr(b)


def div(a, b):
    numerator = _arr(a)
    denominator = _arr(b)
    safe_denominator = np.where(np.abs(denominator) < EPS, np.nan, denominator)
    out = numerator / safe_denominator
    return float(out) if np.ndim(out) == 0 else out


def neg(x):
    out = -_arr(x)
    return float(out) if np.ndim(out) == 0 else out


def abs_val(x):
    out = np.abs(_arr(x))
    return float(out) if np.ndim(out) == 0 else out


def tanh(x):
    out = np.tanh(_arr(x))
    return float(out) if np.ndim(out) == 0 else out


def sign(x):
    out = np.sign(_arr(x))
    return float(out) if np.ndim(out) == 0 else out


def corr(a, b) -> float:
    left = _arr(a)
    right = _arr(b)
    n = min(left.size, right.size)
    if n < 2:
        return float(np.nan)

    left = left[-n:]
    right = right[-n:]
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return float(np.nan)

    left = left[mask]
    right = right[mask]
    if np.std(left) <= EPS or np.std(right) <= EPS:
        return float(np.nan)
    return float(np.corrcoef(left, right)[0, 1])


def cov(a, b) -> float:
    left = _arr(a)
    right = _arr(b)
    n = min(left.size, right.size)
    if n < 2:
        return float(np.nan)

    left = left[-n:]
    right = right[-n:]
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return float(np.nan)
    return float(np.cov(left[mask], right[mask], ddof=0)[0, 1])


def zscore(x) -> np.ndarray:
    values = _arr(x)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std <= EPS:
        return np.full_like(values, np.nan, dtype=float)
    return (values - mean) / std


def rank(x) -> np.ndarray:
    values = _arr(x)
    out = np.full(values.shape, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(values))[0]
    if finite_idx.size == 0:
        return out

    order = finite_idx[np.argsort(values[finite_idx], kind="mergesort")]
    out[order] = np.arange(1, order.size + 1, dtype=float)
    return out


OPERATORS = {
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_sum": ts_sum,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "first": first,
    "last": last,
    "diff": diff,
    "normalize": normalize,
    "ema": ema,
    "log": log,
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "neg": neg,
    "abs": abs_val,
    "abs_s": abs_val,
    "tanh": tanh,
    "tanh_s": tanh,
    "sign": sign,
    "sign_s": sign,
    "corr": corr,
    "cov": cov,
    "zscore": zscore,
    "rank": rank,
}


def get_operator(name: str):
    """Return an operator function by name."""

    try:
        return OPERATORS[str(name).strip().lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(OPERATORS))
        raise ValueError(f"unknown operator '{name}'. valid operators: {valid}") from exc
