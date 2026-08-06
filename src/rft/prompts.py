"""Miner prompt construction from seeded tasks."""

from __future__ import annotations

from dataclasses import asdict

from src.dsl.benchmarks import get_benchmark
from src.dsl.operators import OPERATORS
from src.seeds.task_bank import SeedTask


def task_to_prompt_payload(task: SeedTask, *, namespace: str) -> dict:
    """Convert a seeded task into serializable prompt metadata."""

    benchmark = get_benchmark(task.scenario.benchmark)
    fields = tuple(field for field in benchmark.fields if field.startswith(f"{namespace}."))
    return {
        "seed_expr": task.seed_expr,
        "seed_score": task.seed_score,
        "scenario": asdict(task.scenario),
        "window": asdict(task.window),
        "objective": task.objective,
        "namespace": namespace,
        "allowed_fields": fields,
        "allowed_operators": tuple(sorted(OPERATORS)),
    }


def build_miner_messages(task: SeedTask, *, namespace: str, count: int) -> list[dict[str, str]]:
    """Build messages asking a miner LLM to mutate/improve one seed expression."""

    payload = task_to_prompt_payload(task, namespace=namespace)
    system = (
        "You are a miner LLM for quantitative factor discovery. "
        "Mutate the provided seed expression into valid Factor DSL candidates. "
        "Use only the allowed namespace fields and operators. Return expressions only."
    )
    user = f"""
Seed expression:
{payload["seed_expr"]}

Seed score:
{payload["seed_score"]}

Scenario:
- benchmark: {payload["scenario"]["benchmark"]}
- market: {payload["scenario"]["market"]}
- horizon: {payload["scenario"]["horizon"]}
- objective: {payload["objective"]}

Evaluation window:
- start: {payload["window"]["start"]}
- end: {payload["window"]["end"]}

Allowed namespace:
{payload["namespace"]}

Allowed fields:
{", ".join(payload["allowed_fields"])}

Allowed operators:
{", ".join(payload["allowed_operators"])}

Generate {int(count)} improved or diversified candidate factor expressions.
Output format:
<expr>candidate_expression</expr>
<expr>candidate_expression</expr>

Do not include explanations, markdown fences, imports, assignments, or prose.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
