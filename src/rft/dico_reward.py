"""Diversity-complementarity reward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.rft.discovery_loop import DiscoveryResult


@dataclass(frozen=True)
class DiCoRewardConfig:
    """Weights and thresholds for DiCo reward composition."""

    invalid_reward: float = -1.0
    quality_weight: float = 1.0
    diversity_weight: float = 0.05
    complementarity_weight: float = 0.05
    duplicate_penalty: float = 0.25


@dataclass(frozen=True)
class DiCoRewardResult:
    """Reward breakdown for one candidate."""

    reward: float
    quality: float
    diversity: float
    complementarity: float
    duplicate: bool
    valid: bool


def compute_dico_reward(
    candidate: DiscoveryResult,
    archive,
    *,
    config: DiCoRewardConfig | None = None,
) -> DiCoRewardResult:
    """Compute reward from quality, diversity, and complementarity."""

    cfg = config or DiCoRewardConfig()
    if not candidate.valid:
        return DiCoRewardResult(cfg.invalid_reward, 0.0, 0.0, 0.0, False, False)

    duplicate = archive.has_signature(candidate.signature)
    diversity = 0.0 if duplicate else 1.0
    complementarity = 1.0 - max(0.0, archive.max_metric_correlation(candidate.metrics))
    reward = (
        cfg.quality_weight * float(candidate.score)
        + cfg.diversity_weight * diversity
        + cfg.complementarity_weight * complementarity
        - (cfg.duplicate_penalty if duplicate else 0.0)
    )
    reward = float(np.clip(reward, -1.0, 1.0))
    return DiCoRewardResult(
        reward=reward,
        quality=float(candidate.score),
        diversity=diversity,
        complementarity=complementarity,
        duplicate=duplicate,
        valid=True,
    )
