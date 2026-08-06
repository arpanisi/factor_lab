"""Build Verl prompt datasets from seeded factor-mining tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.rft.prompts import build_miner_messages, task_to_prompt_payload
from src.seeds.task_bank import SeedTask


@dataclass(frozen=True)
class VerlDatasetRow:
    """One rule-reward training row for Verl."""

    data_source: str
    prompt: list[dict[str, str]]
    reward_model: dict[str, Any]
    extra_info: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def build_verl_prompt_rows(
    tasks: Iterable[SeedTask],
    *,
    namespace: str,
    repeats: int = 1,
    data_source: str = "factor_lab/factor_dsl",
    bar_minutes: int = 1440,
) -> list[VerlDatasetRow]:
    """Convert seeded tasks into Verl parquet rows.

    This mirrors the paper's task-bank idea: each row specifies a seed, market
    scenario, evaluation window, and objective. Rewards remain rule-based and
    executable; no oracle LLM is called from these rows during GRPO.
    """

    rows: list[VerlDatasetRow] = []
    repeat_count = max(1, int(repeats))
    for task_index, task in enumerate(tasks):
        prompt = build_miner_messages(task, namespace=namespace, count=1)
        payload = task_to_prompt_payload(task, namespace=namespace)
        family = f"{namespace}_{task.scenario.benchmark}"
        time_split = f"{task.window.start}_{task.window.end}"
        extra_info = {
            "index": task_index,
            "task_id": f"{task.scenario.name}__{family}__{time_split}__{task_index}",
            "seed_id": f"seed_{task_index}",
            "seed_name": f"{namespace}_seed_{task_index}",
            "seed_expr": task.seed_expr,
            "seed_score": float(task.seed_score),
            "family": family,
            "time_split": time_split,
            "start_date": task.window.start,
            "end_date": task.window.end,
            "bar_minutes": int(bar_minutes),
            "namespace": namespace,
            "reward_kind": task.scenario.benchmark,
            "horizon": int(task.scenario.horizon),
            "horizon_bars": int(task.scenario.horizon),
            "window": payload["window"],
            "scenario": payload["scenario"],
            "objective": task.objective,
        }
        for repeat_index in range(repeat_count):
            row_info = dict(extra_info)
            row_info["repeat_index"] = repeat_index
            rows.append(
                VerlDatasetRow(
                    data_source=data_source,
                    prompt=prompt,
                    reward_model={"ground_truth": [], "style": "rule"},
                    extra_info=row_info,
                )
            )
    return rows


def write_verl_prompt_dataset(rows: Iterable[VerlDatasetRow], output_path: Path) -> Path:
    """Write rows to a parquet file expected by Verl-style rule rewards."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [row.to_record() for row in rows]
    if not records:
        raise ValueError("cannot write an empty Verl prompt dataset")
    try:
        pd.DataFrame.from_records(records).to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Writing Verl prompt datasets requires pyarrow or fastparquet. "
            "Install the project requirements on the training server; Vast logs already showed pyarrow installed."
        ) from exc
    return output_path
