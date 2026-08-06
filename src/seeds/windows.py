"""Evaluation-window construction."""

from __future__ import annotations

import pandas as pd

from src.seeds.task_bank import EvaluationWindow


def make_time_windows(
    start: str,
    end: str,
    *,
    periods: int,
) -> tuple[EvaluationWindow, ...]:
    """Split a date range into contiguous evaluation windows."""

    n = int(periods)
    if n <= 0:
        raise ValueError("periods must be positive")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")

    edges = pd.date_range(start_ts, end_ts, periods=n + 1)
    windows = []
    for i in range(n):
        window_start = edges[i]
        window_end = edges[i + 1]
        if i > 0:
            window_start = window_start + pd.Timedelta(1, unit="D")
        windows.append(EvaluationWindow(str(window_start.date()), str(window_end.date())))
    return tuple(windows)
