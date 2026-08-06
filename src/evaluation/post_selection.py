"""Post-selection and equal-weight factor fusion.

This implements the evaluation protocol described in the paper section:
individual factor evaluation, validation ranking, decorrelation filtering,
and equal-weight fusion into a multi-factor signal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.scoring.series import evaluate_factor_series, forward_returns


@dataclass(frozen=True)
class PostSelectionConfig:
    """Configuration for validation-guided decorrelated factor selection."""

    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    correlation_threshold: float = 0.7
    top_k: int = 5
    min_history: int = 30
    min_assets: int = 3
    horizon: int = 1


@dataclass(frozen=True)
class FactorEvaluation:
    """Metrics for one factor on a chosen evaluation split."""

    expr: str
    split: str
    dir_acc: float
    ic_mean: float
    rank_ic_mean: float
    icir: float
    long_short_mean: float
    long_short_sharpe: float
    valid_times: int
    mean_assets_per_time: float

    @property
    def validation_score(self) -> float:
        """Default ranking score used by the paper-style selection protocol."""

        return float(self.rank_ic_mean)


@dataclass(frozen=True)
class FusedEvaluation:
    """Selected factor set plus fused-signal metrics."""

    selected_exprs: tuple[str, ...]
    validation: FactorEvaluation
    test: FactorEvaluation
    correlation_threshold: float


def evaluate_factor_library(
    exprs: Sequence[str],
    namespace: str,
    frames_by_asset: Mapping[str, pd.DataFrame],
    *,
    price_col: str,
    config: PostSelectionConfig | None = None,
) -> FusedEvaluation:
    """Evaluate, select, decorrelate, and fuse a candidate factor library."""

    cfg = config or PostSelectionConfig()
    factor_matrices = {
        expr: _factor_matrix(expr, namespace, frames_by_asset, min_history=cfg.min_history) for expr in exprs
    }
    fwd = _forward_return_matrix(frames_by_asset, price_col=price_col, horizon=cfg.horizon)

    validation_evals = [
        evaluate_factor_matrix(expr, matrix, fwd, split="validation", start=cfg.validation_start, end=cfg.validation_end)
        for expr, matrix in factor_matrices.items()
    ]
    selected = select_decorrelated_factors(
        validation_evals,
        factor_matrices,
        start=cfg.validation_start,
        end=cfg.validation_end,
        threshold=cfg.correlation_threshold,
        top_k=cfg.top_k,
    )

    selected_matrices = [factor_matrices[expr] for expr in selected]
    validation = evaluate_fused_signal(
        selected_matrices,
        fwd,
        split="validation_fused",
        start=cfg.validation_start,
        end=cfg.validation_end,
    )
    test = evaluate_fused_signal(
        selected_matrices,
        fwd,
        split="test_fused",
        start=cfg.test_start,
        end=cfg.test_end,
    )
    return FusedEvaluation(tuple(selected), validation, test, float(cfg.correlation_threshold))


def evaluate_factor_matrix(
    expr: str,
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    *,
    split: str,
    start: str | None = None,
    end: str | None = None,
) -> FactorEvaluation:
    """Compute DirAcc, IC, RankIC, ICIR, and long-short spread for a factor matrix."""

    factor, fwd = _align_split(factor, fwd, start=start, end=end)
    dir_hits: list[float] = []
    ics: list[float] = []
    rank_ics: list[float] = []
    long_short: list[float] = []
    assets_per_time: list[int] = []

    for timestamp in factor.index:
        pair = pd.concat({"factor": factor.loc[timestamp], "fwd": fwd.loc[timestamp]}, axis=1).dropna()
        if pair.shape[0] < 2:
            continue
        assets_per_time.append(int(pair.shape[0]))
        signs = np.sign(pair["factor"].to_numpy(dtype=float))
        fwd_signs = np.sign(pair["fwd"].to_numpy(dtype=float))
        active = signs != 0
        if active.any():
            dir_hits.append(float(np.mean(signs[active] == fwd_signs[active])))
        ics.append(_corr(pair["factor"], pair["fwd"]))
        rank_ics.append(_corr(pair["factor"].rank(method="average"), pair["fwd"].rank(method="average")))
        long_short.append(_long_short_spread(pair["factor"], pair["fwd"]))

    ic_arr = _finite_array(ics)
    rank_arr = _finite_array(rank_ics)
    ls_arr = _finite_array(long_short)
    dir_arr = _finite_array(dir_hits)
    ic_mean = float(np.mean(ic_arr)) if ic_arr.size else float("nan")
    rank_ic_mean = float(np.mean(rank_arr)) if rank_arr.size else float("nan")
    icir = float(ic_mean / (np.std(ic_arr) + 1e-12)) if ic_arr.size else float("nan")
    ls_mean = float(np.mean(ls_arr)) if ls_arr.size else float("nan")
    ls_sharpe = float(ls_mean / (np.std(ls_arr) + 1e-12) * np.sqrt(ls_arr.size)) if ls_arr.size else float("nan")
    dir_acc = float(np.mean(dir_arr)) if dir_arr.size else float("nan")

    return FactorEvaluation(
        expr=expr,
        split=split,
        dir_acc=dir_acc,
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic_mean,
        icir=icir,
        long_short_mean=ls_mean,
        long_short_sharpe=ls_sharpe,
        valid_times=int(rank_arr.size),
        mean_assets_per_time=float(np.mean(assets_per_time)) if assets_per_time else 0.0,
    )


def select_decorrelated_factors(
    evaluations: Sequence[FactorEvaluation],
    factor_matrices: Mapping[str, pd.DataFrame],
    *,
    start: str | None = None,
    end: str | None = None,
    threshold: float = 0.7,
    top_k: int = 5,
) -> list[str]:
    """Rank by validation RankIC and greedily keep factors below a correlation threshold."""

    ranked = sorted(evaluations, key=lambda item: _nan_to_floor(item.validation_score), reverse=True)
    selected: list[str] = []
    for evaluation in ranked:
        if len(selected) >= int(top_k):
            break
        candidate = factor_matrices[evaluation.expr]
        max_corr = 0.0
        for expr in selected:
            corr = _matrix_corr(candidate, factor_matrices[expr], start=start, end=end)
            max_corr = max(max_corr, abs(corr) if np.isfinite(corr) else 0.0)
        if max_corr <= float(threshold):
            selected.append(evaluation.expr)
    return selected


def evaluate_fused_signal(
    factor_matrices: Sequence[pd.DataFrame],
    fwd: pd.DataFrame,
    *,
    split: str,
    start: str | None = None,
    end: str | None = None,
) -> FactorEvaluation:
    """Equal-weight normalized-rank fusion of selected factor matrices."""

    if not factor_matrices:
        empty = pd.DataFrame(index=fwd.index, columns=fwd.columns, dtype=float)
        return evaluate_factor_matrix("equal_weight_fusion", empty, fwd, split=split, start=start, end=end)
    aligned = [_rank_normalize(matrix) for matrix in factor_matrices]
    fused = sum(aligned) / float(len(aligned))
    return evaluate_factor_matrix("equal_weight_fusion", fused, fwd, split=split, start=start, end=end)


def write_evaluation_report(path: Path, evaluation: FusedEvaluation) -> None:
    """Write a compact JSON report."""

    payload = {
        "selected_exprs": list(evaluation.selected_exprs),
        "correlation_threshold": evaluation.correlation_threshold,
        "validation": asdict(evaluation.validation),
        "test": asdict(evaluation.test),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _factor_matrix(
    expr: str,
    namespace: str,
    frames_by_asset: Mapping[str, pd.DataFrame],
    *,
    min_history: int,
) -> pd.DataFrame:
    series = {
        str(asset): evaluate_factor_series(expr, namespace, frame, min_history=min_history)
        for asset, frame in frames_by_asset.items()
    }
    return pd.concat(series, axis=1)


def _forward_return_matrix(
    frames_by_asset: Mapping[str, pd.DataFrame],
    *,
    price_col: str,
    horizon: int,
) -> pd.DataFrame:
    return pd.concat(
        {str(asset): forward_returns(frame, price_col=price_col, horizon=horizon) for asset, frame in frames_by_asset.items()},
        axis=1,
    )


def _align_split(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = factor.index.intersection(fwd.index)
    common_cols = factor.columns.intersection(fwd.columns)
    factor = factor.loc[common_index, common_cols]
    fwd = fwd.loc[common_index, common_cols]
    if start:
        start_ts = pd.Timestamp(start)
        factor = factor.loc[factor.index >= start_ts]
        fwd = fwd.loc[fwd.index >= start_ts]
    if end:
        end_ts = pd.Timestamp(end)
        factor = factor.loc[factor.index <= end_ts]
        fwd = fwd.loc[fwd.index <= end_ts]
    return factor, fwd


def _rank_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    ranked = matrix.rank(axis=1, method="average", pct=True)
    return (ranked - 0.5) * 2.0


def _matrix_corr(left: pd.DataFrame, right: pd.DataFrame, *, start: str | None, end: str | None) -> float:
    left, right = _align_split(left, right, start=start, end=end)
    joined = pd.concat({"left": left.stack(), "right": right.stack()}, axis=1).dropna()
    if joined.shape[0] < 2:
        return 0.0
    return _corr(joined["left"], joined["right"])


def _corr(left: pd.Series, right: pd.Series) -> float:
    if left.shape[0] < 2:
        return float("nan")
    if float(left.std(ddof=0)) <= 1e-12 or float(right.std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(left.corr(right))


def _long_short_spread(factor: pd.Series, fwd: pd.Series) -> float:
    n = max(1, int(np.floor(factor.shape[0] * 0.2)))
    ordered = factor.sort_values()
    return float(fwd.loc[ordered.index[-n:]].mean() - fwd.loc[ordered.index[:n]].mean())


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _nan_to_floor(value: float) -> float:
    return float(value) if np.isfinite(value) else -1e12
