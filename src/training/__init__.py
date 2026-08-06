"""Training scaffolds for GRPO factor discovery."""

from src.training.grpo_reward import FactorRewardRuntime, make_grpo_reward_func

__all__ = [
    "FactorRewardRuntime",
    "make_grpo_reward_func",
]
