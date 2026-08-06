"""Mined factor database."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.rft.discovery_loop import DiscoveryResult
from src.rft.dico_reward import DiCoRewardResult


@dataclass(frozen=True)
class MinedFactorRecord:
    """Stored validated factor."""

    expr: str
    signature: str
    score: float
    reward: float
    metrics: dict[str, float]
    task: dict


@dataclass(frozen=True)
class DatabaseSelectionConfig:
    """Acceptance thresholds for storing mined factors."""

    min_score: float = 0.0
    min_reward: float = 0.0


class MinedFactorDatabase:
    """In-memory mined factor archive with JSONL persistence."""

    def __init__(self, records: tuple[MinedFactorRecord, ...] | None = None):
        self._records: list[MinedFactorRecord] = list(records or ())
        self._signatures = {record.signature for record in self._records}

    @property
    def records(self) -> tuple[MinedFactorRecord, ...]:
        return tuple(self._records)

    def has_signature(self, signature: str) -> bool:
        return bool(signature) and signature in self._signatures

    def maybe_add(
        self,
        candidate: DiscoveryResult,
        reward: DiCoRewardResult,
        *,
        task_payload: dict,
        config: DatabaseSelectionConfig | None = None,
    ) -> bool:
        """Store a candidate if valid, high-quality, and non-duplicate."""

        cfg = config or DatabaseSelectionConfig()
        if not candidate.valid:
            return False
        if self.has_signature(candidate.signature):
            return False
        if candidate.score < cfg.min_score or reward.reward < cfg.min_reward:
            return False

        record = MinedFactorRecord(
            expr=candidate.expr,
            signature=candidate.signature,
            score=float(candidate.score),
            reward=float(reward.reward),
            metrics=dict(candidate.metrics),
            task=task_payload,
        )
        self._records.append(record)
        self._signatures.add(record.signature)
        return True

    def max_metric_correlation(self, metrics: dict[str, float]) -> float:
        """Return max absolute correlation against archived metric profiles."""

        if not self._records:
            return 0.0

        values = []
        for record in self._records:
            corr = _metric_correlation(metrics, record.metrics)
            if np.isfinite(corr):
                values.append(abs(float(corr)))
        return max(values) if values else 0.0

    def save_jsonl(self, path: Path) -> None:
        """Persist records as JSONL."""

        with path.open("w") as f:
            for record in self._records:
                f.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> "MinedFactorDatabase":
        records = []
        if not path.exists():
            return cls()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            records.append(MinedFactorRecord(**json.loads(line)))
        return cls(tuple(records))


def _metric_correlation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left).intersection(right))
    if len(keys) < 2:
        return 0.0
    x = np.asarray([left[key] for key in keys], dtype=float)
    y = np.asarray([right[key] for key in keys], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
