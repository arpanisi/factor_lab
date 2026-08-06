from src.verl_integration.verl_main import build_verl_overrides, configure_reward_environment


def _config():
    return {
        "experiment_name": "demo",
        "model": {
            "name_or_path": "/workspace/models/Qwen3-14B",
            "lora_rank": 0,
            "trust_remote_code": True,
            "gradient_checkpointing": True,
            "remove_padding": True,
        },
        "data": {
            "train_parquet": "train.parquet",
            "val_parquet": "val.parquet",
            "crypto_panel": "data/crypto_panel_clean.pkl",
            "tickers": "BTC-USD,ETH-USD",
            "max_prompt_length": 1536,
            "max_response_length": 256,
        },
        "reward": {
            "archive_jsonl": "factor_lab/outputs/verl/archive.jsonl",
            "reward_log_jsonl": "factor_lab/outputs/verl/reward.jsonl",
        },
        "verl": {
            "actor_rollout_ref": {
                "rollout": {"n": 8, "temperature": 1.0},
                "actor": {
                    "ppo_mini_batch_size": 256,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "optim": {"lr": 5e-6},
                },
            },
            "trainer": {"n_gpus_per_node": 8, "save_freq": 20},
        },
    }


def test_build_verl_overrides_preserves_full_ft_and_a100_scale():
    overrides = build_verl_overrides(_config())

    assert overrides["actor_rollout_ref"]["model"]["path"] == "/workspace/models/Qwen3-14B"
    assert overrides["actor_rollout_ref"]["model"]["lora_rank"] == 0
    assert overrides["trainer"]["n_gpus_per_node"] == 8
    assert overrides["actor_rollout_ref"]["rollout"]["n"] == 8
    assert overrides["data"]["train_files"] == ["train.parquet"]
    assert overrides["data"]["val_files"] == ["val.parquet"]
    assert overrides["algorithm"]["adv_estimator"] == "grpo"


def test_configure_reward_environment_sets_worker_paths(monkeypatch):
    configure_reward_environment(_config())

    import os

    assert os.environ["FACTOR_LAB_CRYPTO_PANEL"] == "data/crypto_panel_clean.pkl"
    assert os.environ["FACTOR_LAB_TICKERS"] == "BTC-USD,ETH-USD"
    assert os.environ["FACTOR_LAB_ARCHIVE_JSONL"] == "factor_lab/outputs/verl/archive.jsonl"
    assert os.environ["FACTOR_LAB_REWARD_LOG_JSONL"] == "factor_lab/outputs/verl/reward.jsonl"
