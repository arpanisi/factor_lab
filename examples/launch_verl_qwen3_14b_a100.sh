#!/usr/bin/env bash
set -euo pipefail

# Prep step for the paper-scale (8x A100, full-parameter, no LoRA) GRPO run.
# Run this from inside factor_lab/. It does not launch training itself - it
# only (1) builds the train/val task-bank parquet files Verl reads as its
# dataset, and (2) prints the exact command to launch Verl afterward.

# Env vars consumed by the reward workers at actual training time (set here,
# with a fallback default, so build_dataset below and the eventual Verl
# reward_fn agree on the same crypto data/tickers/output locations).
export FACTOR_LAB_CRYPTO_PANEL="${FACTOR_LAB_CRYPTO_PANEL:-data/crypto/crypto_panel_clean.pkl}"
export FACTOR_LAB_TICKERS="${FACTOR_LAB_TICKERS:-ADA-USD,BNB-USD,BTC-USD,DOGE-USD,ETH-USD,LINK-USD,XLM-USD,XRP-USD}"
export FACTOR_LAB_ARCHIVE_JSONL="${FACTOR_LAB_ARCHIVE_JSONL:-outputs/verl/qwen3_14b_fullft/mined_factors.jsonl}"
export FACTOR_LAB_REWARD_LOG_JSONL="${FACTOR_LAB_REWARD_LOG_JSONL:-outputs/verl/qwen3_14b_fullft/reward_rollouts.jsonl}"

# Build the TRAINING task-bank parquet: one seed expression x one scenario,
# repeated 400x into prompt rows so Verl has enough rows for a GRPO epoch.
python -m src.verl_integration.build_dataset \
  --output outputs/verl/crypto_grpo_tasks.parquet \
  --seed-expr "div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))" \
  --seed-score 0.38385972330719526 \
  --crypto-panel "${FACTOR_LAB_CRYPTO_PANEL}" \
  --tickers "${FACTOR_LAB_TICKERS}" \
  --repeats 400

# Build the VALIDATION task-bank parquet: same seed/scenario, far fewer
# repeats (32) since it's only used for periodic eval, not gradient updates.
python -m src.verl_integration.build_dataset \
  --output outputs/verl/crypto_grpo_val_tasks.parquet \
  --seed-expr "div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))" \
  --seed-score 0.38385972330719526 \
  --crypto-panel "${FACTOR_LAB_CRYPTO_PANEL}" \
  --tickers "${FACTOR_LAB_TICKERS}" \
  --repeats 32

# This just prints a copy-paste reference for the operator - it does not run
# anything. The actual Verl launch is a separate, long-running GPU command.
cat <<'MSG'
Dataset and reward bridge are ready for Verl.

Primary paper-scale command:

  python -m src.verl_integration.verl_main \
    --config config/verl_qwen3_14b_fullft_a100.yaml \
    --base-config config/verl_ppo_trainer_base.yaml

Key values:
  train parquet: outputs/verl/crypto_grpo_tasks.parquet
  val parquet:   outputs/verl/crypto_grpo_val_tasks.parquet
  reward bridge: src.verl_integration.reward_bridge.FactorLabVerlRewardBridge
  model:         /workspace/models/Qwen3-14B
  GPUs:          8
  lora_rank:     0
MSG
