"""QLoRA GRPO entrypoint for executable factor-reward training.

This is designed for a GPU server. It wires TRL GRPO to the existing
DSL-realization, backtest-oracle, DiCo reward, and mined-factor archive.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pandas as pd

from examples.build_seed_bank import crypto_frames_from_panel
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL
from src.rft import MinedFactorDatabase
from src.seeds import EvaluationWindow, FactorScenario, SeedCandidate, build_task_bank
from src.training.grpo_reward import FactorRewardRuntime, make_grpo_reward_func


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QLoRA GRPO for Factor DSL reward optimization.")
    parser.add_argument("--model", required=True, help="Base or SFT adapter/model path.")
    parser.add_argument("--seed-expr", required=True)
    parser.add_argument("--seed-score", type=float, default=0.0)
    parser.add_argument("--crypto-panel", type=Path, default=DEFAULT_CRYPTO_PANEL)
    parser.add_argument("--tickers", default="ADA-USD,BNB-USD,BTC-USD,DOGE-USD,ETH-USD,LINK-USD,XLM-USD,XRP-USD")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-jsonl", type=Path, default=Path("factor_lab/outputs/grpo/mined_factors.jsonl"))
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--dataset-repeat", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--loss-type", default="dapo", help="Use dapo when supported by installed TRL.")
    parser.add_argument("--reward-log-jsonl", type=Path)
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer
    import torch

    panel = pd.read_pickle(args.crypto_panel)
    tickers = tuple(item.strip() for item in args.tickers.split(",") if item.strip()) or None
    frames = crypto_frames_from_panel(panel, tickers=tickers)
    close = panel["close"]
    scenario = FactorScenario.from_benchmark(
        "daily_cross_sectional_rankic",
        market="crypto",
        horizon=1,
        name="crypto_grpo_factor_discovery",
    )
    task = build_task_bank(
        [SeedCandidate(args.seed_expr, float(args.seed_score), "")],
        scenario,
        [EvaluationWindow(str(close.index.min().date()), str(close.index.max().date()))],
    )[0]
    from src.rft.prompts import build_miner_messages

    prompt = build_miner_messages(task, namespace="crypto", count=1)
    dataset_size = (
        int(args.dataset_repeat)
        if int(args.dataset_repeat) > 0
        else max(8, int(args.steps) * int(args.batch_size) * int(args.grad_accum))
    )
    dataset = Dataset.from_list([{"prompt": prompt} for _ in range(dataset_size)])
    archive = MinedFactorDatabase.load_jsonl(args.archive_jsonl)
    reward_func = make_grpo_reward_func(
        FactorRewardRuntime(
            task=task,
            namespace="crypto",
            data=frames,
            price_col="close",
            archive=archive,
            min_history=30,
            min_assets=8,
            reward_log_jsonl=args.reward_log_jsonl,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    grpo_kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.lr,
        "max_steps": args.steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_generations": args.generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "beta": args.beta,
        "bf16": True,
        "logging_steps": 1,
        "save_steps": args.save_steps,
        "report_to": "none",
    }
    grpo_args = _build_grpo_config(GRPOConfig, grpo_kwargs, loss_type=args.loss_type)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    args.archive_jsonl.parent.mkdir(parents=True, exist_ok=True)
    archive.save_jsonl(args.archive_jsonl)
    return 0


def _build_grpo_config(grpo_config_cls, kwargs: dict, *, loss_type: str):
    """Build GRPOConfig using only keys supported by installed TRL."""

    signature = inspect.signature(grpo_config_cls)
    supported = set(signature.parameters)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    if "loss_type" in supported:
        filtered["loss_type"] = loss_type
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"GRPOConfig: dropped unsupported args: {', '.join(dropped)}")
    return grpo_config_cls(**filtered)


if __name__ == "__main__":
    raise SystemExit(main())
