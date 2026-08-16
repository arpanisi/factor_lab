"""Factor Lab pure-Verl GRPO launcher.

This mirrors QuantEvolver's pure-Verl pattern while using Factor Lab's WRDS /
crypto data adapters, task parquet, and executable reward bridge.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def load_factor_verl_config(path: str | Path) -> dict[str, Any]:
    """Load the Factor Lab Verl YAML config."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency check.
        raise RuntimeError("PyYAML is required to load Factor Lab Verl configs") from exc

    with Path(path).open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping config at {path}")
    return data


def build_verl_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Build Verl override config from the Factor Lab run config."""

    model = config.get("model", {})
    data = config.get("data", {})
    reward = config.get("reward", {})
    verl = config.get("verl", {})
    actor_rollout_ref = verl.get("actor_rollout_ref", {})
    actor = actor_rollout_ref.get("actor", {})
    rollout = actor_rollout_ref.get("rollout", {})
    trainer = verl.get("trainer", {})
    algorithm = verl.get("algorithm", {})

    train_file = str(data["train_parquet"])
    val_file = str(data.get("val_parquet") or train_file)
    output_root = str(Path(str(reward.get("archive_jsonl", "outputs/verl/mined_factors.jsonl"))).parent)
    experiment_name = str(config.get("experiment_name") or "factor_lab_qwen3_14b_fullft_grpo")

    return {
        "algorithm": {
            "adv_estimator": algorithm.get("adv_estimator", "grpo"),
            "use_kl_in_reward": bool(algorithm.get("use_kl_in_reward", False)),
        },
        "data": {
            "train_files": [train_file],
            "val_files": [val_file],
            "train_batch_size": int(data.get("train_batch_size", actor.get("ppo_mini_batch_size", 256))),
            "val_batch_size": int(data.get("val_batch_size", actor.get("ppo_mini_batch_size", 256))),
            "max_prompt_length": int(data.get("max_prompt_length", 1536)),
            "max_response_length": int(data.get("max_response_length", 256)),
            "return_raw_chat": False,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "dataloader_num_workers": int(data.get("dataloader_num_workers", 0)),
            "validation_shuffle": False,
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(model["name_or_path"]),
                "use_remove_padding": bool(model.get("remove_padding", True)),
                "enable_gradient_checkpointing": bool(model.get("gradient_checkpointing", True)),
                "trust_remote_code": bool(model.get("trust_remote_code", True)),
                "lora_rank": int(model.get("lora_rank", 0)),
            },
            "rollout": {
                "name": rollout.get("name", "vllm"),
                "mode": rollout.get("mode", "sync"),
                "n": int(rollout.get("n", 8)),
                "temperature": float(rollout.get("temperature", 1.0)),
                "top_p": float(rollout.get("top_p", 0.95)),
                "top_k": int(rollout.get("top_k", 50)),
                "prompt_length": int(data.get("max_prompt_length", 1536)),
                "response_length": int(data.get("max_response_length", 256)),
                "max_model_len": int(rollout.get("max_model_len", 2048)),
                "gpu_memory_utilization": float(rollout.get("gpu_memory_utilization", 0.80)),
                "tensor_model_parallel_size": int(rollout.get("tensor_model_parallel_size", 1)),
                "log_prob_micro_batch_size_per_gpu": int(rollout.get("log_prob_micro_batch_size_per_gpu", 1)),
                "log_prob_max_token_len_per_gpu": int(rollout.get("log_prob_max_token_len_per_gpu", 8000)),
                "val_kwargs": {"n": int(rollout.get("val_n", 1)), "do_sample": False},
            },
            "actor": {
                "strategy": actor.get("strategy", "fsdp"),
                "ppo_mini_batch_size": int(actor.get("ppo_mini_batch_size", 256)),
                "ppo_micro_batch_size_per_gpu": int(actor.get("ppo_micro_batch_size_per_gpu", 1)),
                "use_kl_loss": bool(actor.get("use_kl_loss", True)),
                "kl_loss_coef": float(actor.get("kl_loss_coef", 0.001)),
                "kl_loss_type": actor.get("kl_loss_type", "low_var_kl"),
                "entropy_coeff": float(actor.get("entropy_coeff", 0)),
                "optim": {"lr": float(actor.get("optim", {}).get("lr", 5e-6))},
                "fsdp_config": actor.get(
                    "fsdp_config",
                    {"param_offload": False, "optimizer_offload": True},
                ),
                "ppo_max_token_len_per_gpu": int(actor.get("ppo_max_token_len_per_gpu", 8000)),
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": int(rollout.get("log_prob_micro_batch_size_per_gpu", 1)),
                "log_prob_max_token_len_per_gpu": int(rollout.get("log_prob_max_token_len_per_gpu", 8000)),
                "fsdp_config": {"param_offload": True},
            },
        },
        "critic": {
            "strategy": "fsdp",
            "ppo_max_token_len_per_gpu": int(actor.get("ppo_max_token_len_per_gpu", 8000)),
            "forward_max_token_len_per_gpu": int(actor.get("ppo_max_token_len_per_gpu", 8000)),
        },
        "reward_model": {"enable": False, "launch_reward_fn_async": False},
        "trainer": {
            "project_name": str(trainer.get("project_name", "factor_lab")),
            "experiment_name": experiment_name,
            "logger": trainer.get("logger", ["console"]),
            "nnodes": int(trainer.get("nnodes", 1)),
            "n_gpus_per_node": int(trainer.get("n_gpus_per_node", 8)),
            "total_epochs": int(trainer.get("total_epochs", 1)),
            "save_freq": int(trainer.get("save_freq", 20)),
            "test_freq": int(trainer.get("test_freq", trainer.get("save_freq", 20))),
            "val_before_train": bool(trainer.get("val_before_train", True)),
            "critic_warmup": 0,
            "default_local_dir": str(Path(output_root) / "checkpoints" / experiment_name),
            "validation_data_dir": str(Path(output_root) / "validation_log"),
            "rollout_data_dir": str(Path(output_root) / "rollout_log"),
        },
        "ray_init": {"num_cpus": trainer.get("num_cpus")},
    }


def build_verl_config(config: dict[str, Any], *, base_config: str | Path):
    """Merge Factor Lab overrides into QuantEvolver/Verl's base YAML."""

    from omegaconf import OmegaConf

    base = OmegaConf.load(base_config)
    overrides = OmegaConf.create(build_verl_overrides(config))
    merged = OmegaConf.merge(base, overrides)
    OmegaConf.resolve(merged)
    return merged


def configure_reward_environment(config: dict[str, Any]) -> None:
    """Set env vars consumed by reward workers."""

    data = config.get("data", {})
    reward = config.get("reward", {})
    env_values = {
        "FACTOR_LAB_CRYPTO_PANEL": data.get("crypto_panel"),
        "FACTOR_LAB_TICKERS": data.get("tickers"),
        "FACTOR_LAB_ARCHIVE_JSONL": reward.get("archive_jsonl"),
        "FACTOR_LAB_REWARD_LOG_JSONL": reward.get("reward_log_jsonl"),
    }
    for key, value in env_values.items():
        if value is not None:
            os.environ[key] = str(value)


def run(config_path: str | Path, *, base_config: str | Path):
    """Run distributed GRPO through Verl."""

    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    from src.verl_integration.reward_bridge import FactorLabVerlRewardBridge

    raw_config = load_factor_verl_config(config_path)
    configure_reward_environment(raw_config)
    cfg = build_verl_config(raw_config, base_config=base_config)
    print(OmegaConf.to_yaml(cfg))

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_USE_V1": "1",
                    "FACTOR_LAB_CRYPTO_PANEL": os.environ.get("FACTOR_LAB_CRYPTO_PANEL", ""),
                    "FACTOR_LAB_TICKERS": os.environ.get("FACTOR_LAB_TICKERS", ""),
                    "FACTOR_LAB_ARCHIVE_JSONL": os.environ.get("FACTOR_LAB_ARCHIVE_JSONL", ""),
                    "FACTOR_LAB_REWARD_LOG_JSONL": os.environ.get("FACTOR_LAB_REWARD_LOG_JSONL", ""),
                }
            },
            num_cpus=cfg.ray_init.num_cpus,
        )

    local_path = copy_to_local(
        cfg.actor_rollout_ref.model.path,
        use_shm=cfg.actor_rollout_ref.model.get("use_shm", False),
    )
    tokenizer = hf_tokenizer(local_path, trust_remote_code=cfg.data.get("trust_remote_code", True))
    processor = hf_processor(local_path, trust_remote_code=cfg.data.get("trust_remote_code", True), use_fast=True)

    if cfg.actor_rollout_ref.actor.strategy not in ["fsdp", "fsdp2"]:
        raise NotImplementedError(cfg.actor_rollout_ref.actor.strategy)

    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    mapping = {Role.ActorRollout: "global_pool"}
    if cfg.actor_rollout_ref.actor.use_kl_loss or cfg.algorithm.use_kl_in_reward:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={"global_pool": [cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes},
        mapping=mapping,
    )

    train_dataset = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer, processor)
    val_dataset = create_rl_dataset(cfg.data.val_files, cfg.data, tokenizer, processor)
    train_sampler = create_rl_sampler(cfg.data, train_dataset)
    reward_bridge = FactorLabVerlRewardBridge(tokenizer)
    trainer = RayPPOTrainer(
        config=cfg,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_bridge,
        val_reward_fn=reward_bridge,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=cfg.trainer.device,
    )
    trainer.init_workers()
    trainer.fit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Factor Lab GRPO through pure Verl.")
    parser.add_argument("--config", default="config/verl_qwen3_14b_fullft_a100.yaml")
    parser.add_argument(
        "--base-config",
        default="config/verl_ppo_trainer_base.yaml",
        help="Verl PPO base YAML to merge with Factor Lab overrides.",
    )
    parser.add_argument("--print-config", action="store_true", help="Build and print the merged config without launching.")
    args = parser.parse_args(argv)

    raw_config = load_factor_verl_config(args.config)
    if args.print_config:
        from omegaconf import OmegaConf

        cfg = build_verl_config(raw_config, base_config=args.base_config)
        print(OmegaConf.to_yaml(cfg))
        return 0
    run(args.config, base_config=args.base_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
