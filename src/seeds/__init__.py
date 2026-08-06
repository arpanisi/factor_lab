"""Seed construction tools for Factor Lab."""

from src.seeds.build import ScenarioSeedBuild, build_scenario_seed_bank
from src.seeds.candidates import candidate_templates_for_scenario
from src.seeds.canonical import canonical_signature
from src.seeds.oracle import (
    OpenRouterConfig,
    build_oracle_seed_messages,
    extract_seed_expressions,
    generate_oracle_seed_candidates,
    generate_oracle_seed_candidates_from_file,
)
from src.seeds.pool import SeedCandidate, SeedPoolConfig, build_seed_pool
from src.seeds.refine import RefinedScenario, refine_scenario
from src.seeds.scenario import FactorScenario
from src.seeds.task_bank import EvaluationWindow, SeedTask, build_task_bank
from src.seeds.windows import make_time_windows

__all__ = [
    "EvaluationWindow",
    "FactorScenario",
    "OpenRouterConfig",
    "RefinedScenario",
    "ScenarioSeedBuild",
    "SeedCandidate",
    "SeedPoolConfig",
    "SeedTask",
    "build_seed_pool",
    "build_scenario_seed_bank",
    "build_oracle_seed_messages",
    "build_task_bank",
    "candidate_templates_for_scenario",
    "canonical_signature",
    "extract_seed_expressions",
    "generate_oracle_seed_candidates",
    "generate_oracle_seed_candidates_from_file",
    "make_time_windows",
    "refine_scenario",
]
