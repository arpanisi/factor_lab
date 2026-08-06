"""Local discovery components for factor generation."""

from src.rft.database import DatabaseSelectionConfig, MinedFactorDatabase, MinedFactorRecord
from src.rft.dico_reward import DiCoRewardConfig, DiCoRewardResult, compute_dico_reward
from src.rft.discovery_loop import DiscoveryResult, run_discovery_for_task
from src.rft.miner import MinerConfig, generate_miner_candidates
from src.rft.prompts import build_miner_messages, task_to_prompt_payload
from src.rft.realizer import RealizedFactor, extract_factor_expressions, realize_factor
from src.rft.reward_bridge import RewardBridgeResult, reward_completion, reward_completions

__all__ = [
    "DatabaseSelectionConfig",
    "DiscoveryResult",
    "DiCoRewardConfig",
    "DiCoRewardResult",
    "MinerConfig",
    "MinedFactorDatabase",
    "MinedFactorRecord",
    "RealizedFactor",
    "RewardBridgeResult",
    "build_miner_messages",
    "compute_dico_reward",
    "extract_factor_expressions",
    "generate_miner_candidates",
    "realize_factor",
    "reward_completion",
    "reward_completions",
    "run_discovery_for_task",
    "task_to_prompt_payload",
]
