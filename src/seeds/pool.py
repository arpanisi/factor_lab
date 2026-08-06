"""Seed pool construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from src.dsl.validator import ValidationError, validate_expr
from src.seeds.canonical import canonical_signature


ScoreFn = Callable[[str], float]


@dataclass(frozen=True)
class SeedCandidate:
    """A valid, scored, deduplicated seed expression."""

    expr: str
    score: float
    signature: str


@dataclass(frozen=True)
class SeedPoolConfig:
    """Selection controls for seed pool construction."""

    top_k: int = 16
    quality_threshold: float = float("-inf")


def build_seed_pool(
    raw_candidates: tuple[str, ...] | list[str],
    *,
    scores: Mapping[str, float] | None = None,
    score_fn: ScoreFn | None = None,
    config: SeedPoolConfig | None = None,
) -> tuple[SeedCandidate, ...]:
    """Validate, score, deduplicate, and select TopK seed candidates."""

    cfg = config or SeedPoolConfig()
    scored: list[SeedCandidate] = []

    for expr in raw_candidates:
        expr = str(expr).strip()
        if not expr:
            continue

        try:
            validate_expr(expr)
            signature = canonical_signature(expr)
        except (ValidationError, ValueError):
            continue

        score = _score_expr(expr, scores=scores, score_fn=score_fn)
        if score < cfg.quality_threshold:
            continue
        scored.append(SeedCandidate(expr=expr, score=float(score), signature=signature))

    scored.sort(key=lambda item: item.score, reverse=True)

    selected: list[SeedCandidate] = []
    signatures: set[str] = set()
    for candidate in scored:
        if candidate.signature in signatures:
            continue
        selected.append(candidate)
        signatures.add(candidate.signature)
        if len(selected) >= cfg.top_k:
            break

    return tuple(selected)


def _score_expr(expr: str, *, scores: Mapping[str, float] | None, score_fn: ScoreFn | None) -> float:
    if score_fn is not None:
        return float(score_fn(expr))
    if scores is not None:
        return float(scores.get(expr, 0.0))
    return 0.0
