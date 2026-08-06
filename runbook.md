# Factor Lab GRPO Runbook

This runbook follows the paper path directly: build seeds/task bank first, then run GRPO with executable Factor-DSL rewards. There is no SFT step.

## Repository Layout

```text
config/       Run configs: OpenRouter model roles, GRPO/Verl YAML configs
data/         Local data: WRDS pulls (data/crsp), crypto panels (data/crypto), TAQ bars
docs/         QuantEvolver compatibility notes, coding-plan history, model results
examples/     Runnable CLIs: DSL smoke test, seed-bank builder, baseline rollout, Verl launch script
outputs/      Generated artifacts: task-bank parquet, mined-factor archive, reward logs
src/
  dsl/               The Factor DSL: namespaces, fields, operators, parser, validator, evaluator
  data/              Adapters mapping raw WRDS/crypto data into DSL namespace schemas
  seeds/             Scenario refinement, oracle-LLM seeding, candidate scoring, task-bank construction
  scoring/           RankIC / IC / ICIR / directional-accuracy backtest metrics
  rft/               The local (no-GPU) discovery loop: miner prompts, realizer, DiCo reward, mined-factor database
  training/          Single-GPU TRL/QLoRA GRPO training (debug path)
  verl_integration/  Full-scale GRPO training via Verl + Ray
  evaluation/        Post-selection evaluation of mined factor libraries
  benchmarks/        Comparison harness across candidate-generation approaches
tests/        Unit tests mirroring the src/ layout
```

## Factor DSL Syntax

Every candidate factor is an expression of the form `namespace.field(window)` composed with a small, fixed operator set. Nothing outside this grammar is accepted, so every candidate is mechanically parsed and validated before it is ever executed against data.

```text
namespaces: crsp | taq | crypto
fields:     e.g. crsp.dlyret, crsp.dlyclose, crsp.dlyvol, crsp.dlycap,
            taq.spread, taq.midret, taq.imbalance,
            crypto.returns, crypto.close, crypto.volume
operators:  ts_mean, ts_std, ts_sum, ts_min, ts_max, first, last, diff,
            normalize, ema, log, add, sub, mul, div, neg, abs, tanh,
            sign, corr, cov, zscore, rank
```

Example expression:

```text
div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))
```

The underlying WRDS tables are CRSP `dsf_v2` (daily equity price/volume/return/market-cap) and TAQ intraday trade-and-quote data.

## 1. What To Upload

Upload these from your laptop repo to the Vast instance:

```bash
factor_lab/
ClassProject/data/crypto_panel_clean.pkl
runbook.md
```

Optional, only if you want WRDS access from Vast:

```bash
.env
```

Do not upload `.pgpass` unless you intentionally need WRDS from the server.

## 2. Start On Vast

On the Vast machine:

```bash
cd /workspace
python --version
nvidia-smi
```

Paper-scale target shape:

```text
Python 3.12.13
8x NVIDIA A100
```

The earlier `1x A100 40GB` QLoRA path is only a debug fallback. The paper-scale run should use the Verl dataset/reward path below with full-parameter fine-tuning.

## 3. Create Environment

If the template already has the packages globally, this can still use a venv for cleanliness:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install only the needed stack:

```bash
python -m pip install -r factor_lab/requirements.txt
```

If the template already has compatible GPU builds, do not reinstall torch manually.

Verified package versions from the working install:

```text
accelerate==1.14.0
bitsandbytes==0.50.0
peft==0.20.0
pyarrow==25.0.0
transformers==5.14.1
trl==1.9.2
```

Verify the imports once the packages are installed. 

```bash
python - <<'PY'
import torch, transformers, peft, bitsandbytes, datasets, trl
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
PY
```

## 4. Generate Required Datasets from WRDS

Navigate to the factor_lab folder and run the following commnad:

```
.venv/bin/python src/data/wrds_crsp.py \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --universe sp500 \
  --max-assets 500 \
  --output-parquet data/crsp/wrds_crsp_daily_panel.parquet \
  --output-frames data/crsp/wrds_crsp_daily_frames.pkl
```

The TAQ dataset takes longer than the CRSP dataset. First, try with a smaller smoke test to see if the pipeline is working. 

```
.venv/bin/python -m factor_lab/src/data/wrds_taq \
  --symbols AAPL,MSFT \
  --start-date 2024-01-02 \
  --end-date 2024-01-04 \
  --output-dir data/taq_1m_smoke
```

Expand the list of assests:

```
AAPL, MSFT, NVDA, AMZN, META, GOOG, TSLA, JPM, XOM, UNH, AVGO, LLY, V, MA, HD, PG, COST, MRK,A BBV, CVX
```

## 5. Smoke Test The Project

```bash
python -m pytest tests/test_training_scaffold.py tests/test_reward_bridge.py
```

Expected:

```text
passed
```

## 6. Download Models from Huggingface

Syntax for downloading models individually

```
hf download model-provider/path-to-model --local-dir models/path-to-model
```

Alternately, use the following command to download all the models listed down in 

```
./download_hf_models.sh
```

For smoke test on a single A100, use any of the models downloaded with qLoRA option. For better GRPO training, you will need to run on without LoRA and higher number of rollouts. Subsequently you will need more compute power (GPUs) to execute the training. These runtime codes are given in the later part of the documet

`Qwen2-0.5B` remains useful as a cheap debug model because it produced valid terminated Factor-DSL completions. It is not the full-scale GRPO target.

Observed model results:

| Model | Status | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Works | Produces valid terminated completions and non-`-1` executable rewards. Current best baseline. |
| `/workspace/models/DeepSeek-R1-Distill-Llama-8B` | Failed | Completions hit max length, `clipped_ratio=1`, `mean_terminated_length=0`, reward stayed `-1`, loss/grad stayed `0`. |
| `/workspace/models/Qwen3.5-9B` | Failed | Same failure pattern: max-length unparseable completions, `clipped_ratio=1`, reward `-1`, loss/grad `0`. |

For failed models, stop early if logs show:

```text
completions/clipped_ratio = 1
completions/mean_terminated_length = 0
rewards/reward_func/mean = -1
grad_norm = 0
```

## 7. Pick A Seed

Use one of the current positive seeds from baseline work:

```text
div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))
```

Seed score:

```text
0.655671862964597
```

## 8. Build Verl Task Dataset

This creates the full-scale task bank rows: seed expression, factor scenario, time window, objective, prompt, and rule-reward metadata.

```bash
python -m factor_lab.verl.build_dataset \
  --output factor_lab/outputs/verl/crypto_grpo_tasks.parquet \
  --seed-expr "div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))" \
  --seed-score 0.655671862964597 \
  --crypto-panel data/crypto_panel_clean.pkl \
  --tickers BTC-USD,ETH-USD,XRP-USD \
  --repeats 400
```

Expected:

```text
wrote 400 rows to factor_lab/outputs/verl/crypto_grpo_tasks.parquet
```

## 9. Configure Verl Reward Workers

Set these before launching Verl:

```bash
export FACTOR_LAB_CRYPTO_PANEL=data/crypto_panel_clean.pkl
export FACTOR_LAB_TICKERS=BTC-USD,ETH-USD,XRP-USD
export FACTOR_LAB_ARCHIVE_JSONL=factor_lab/outputs/verl/mined_factors.jsonl
export FACTOR_LAB_REWARD_LOG_JSONL=factor_lab/outputs/verl/reward_rollouts.jsonl
```

Reward function import path:

```text
factor_lab.verl.reward_function.reward_fn
```

This is the executable reward bridge: completions become Factor-DSL expressions, expressions are validated and backtested, RankIC/IC/ICIR-style metrics are converted into DiCo reward, and the reward function returns the scalar tensor used by GRPO.

## 10. Launch Full-Scale Reinforcement Finetuning GRPO With Verl

Use the installed Verl commit's GRPO/PPO launcher and map these values into its config:

```text
train parquet: factor_lab/outputs/verl/crypto_grpo_tasks.parquet
reward bridge: factor_lab.verl.reward_bridge.FactorLabVerlRewardBridge
scalar reward fn: factor_lab.verl.reward_function.reward_fn
model: /workspace/models/Qwen3-14B
GPUs: 8
lora_rank: 0
generations per prompt: 8
max prompt length: 1536
max completion length: 256
temperature: 1.0
top_p: 0.95
top_k: 50
learning_rate: 0.000005
beta: 0.02
save_steps: 20
```

The project-side config lives at:

```text
factor_lab/config/verl_qwen3_14b_fullft_a100.yaml
```

Helper script:

```bash
examples/launch_verl_qwen3_14b_a100.sh
```

That script builds the train/validation parquet files and prints the exact Factor Lab launch command.

Main launch command:

```bash
python -m factor_lab.verl.verl_main \
  --config factor_lab/config/verl_qwen3_14b_fullft_a100.yaml \
  --base-config config/verl_ppo_trainer_base.yaml
```

## 11. Debug Fallback: Single-GPU TRL QLoRA GRPO

Use this only to verify the reward loop before spending on the 8-A100 run.

Conservative A100 40GB run:

```bash
python  training/train_grpo_qlora \
  --model /workspace/models/Qwen2.5-0.5B-Instruct \
  --seed-expr "div(ts_mean(crypto.volume(10)), ts_std(crypto.returns(30)))" \
  --seed-score 0.655671862964597 \
  --crypto-panel data/crypto_panel_clean.pkl \
  --tickers BTC-USD,ETH-USD,XRP-USD \
  --output-dir factor_lab/outputs/grpo/Qwen2.5-0.5B-Instruct_crypto_grpo \
  --archive-jsonl factor_lab/outputs/grpo/Qwen2.5-0.5B-Instruct/mined_factors.jsonl \
  --batch-size 1 \
  --grad-accum 8 \
  --generations 8 \
  --max-prompt-length 1536 \
  --max-completion-length 256 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 50 \
  --repetition-penalty 1.0 \
  --steps 50 \
  --save-steps 20 \
  --dataset-repeat 400 \
  --reward-log-jsonl factor_lab/outputs/grpo/Qwen2.5-7B-Instruct/reward_rollouts.jsonl \
  --lr 0.000005 \
  --beta 0.02 \
  --loss-type dapo
```

## 12. If It OOMs

Reduce in this order:

```bash
--generations 1
--max-prompt-length 1024
--max-completion-length 128
--grad-accum 4
```

## 13. Outputs To Save

After training, save these:

```bash
factor_lab/outputs/verl/crypto_grpo_tasks.parquet
factor_lab/outputs/verl/mined_factors.jsonl
factor_lab/outputs/verl/reward_rollouts.jsonl
factor_lab/outputs/grpo/Qwen2.5-7B-Instruct_crypto_grpo/
factor_lab/outputs/grpo/Qwen2.5-7B-Instruct/mined_factors.jsonl
factor_lab/outputs/grpo/Qwen2.5-7B-Instruct/reward_rollouts.jsonl
```

The model output/checkpoint directory is the trained model artifact. The mined factor JSONL is the accepted factor database. The reward rollout JSONL stores every completion, extracted expression, reward, validity flag, metrics, and rejection reason.

With the default `--save-steps 20`, intermediate checkpoints appear under the output directory as:

```text
checkpoint-20/
checkpoint-40/
```

## 14. What This Run Demonstrates

The miner LLM generates Factor-DSL expressions. Each completion is validated, executed on market data, scored by RankIC/IC/ICIR-style backtest metrics, converted into a scalar reward, and passed directly to GRPO as the policy-gradient reward signal.
