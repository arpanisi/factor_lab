"""Oracle LLM seed generation through OpenRouter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.dsl.benchmarks import get_benchmark
from src.dsl.operators import OPERATORS
from src.dsl.parser import CallNode, ExprNode, FieldNode, NumberNode, parse_expr
from src.dsl.validator import validate_expr
from src.llm.openrouter import call_openrouter_chat, openrouter_content
from src.seeds.scenario import FactorScenario


_EXPR_RE = re.compile(r"<expr>\s*(.*?)\s*</expr>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenRouter request settings for oracle seed generation."""

    model: str = "deepseek/deepseek-chat-v3.1"
    temperature: float = 0.7
    max_tokens: int = 1600
    timeout_seconds: int = 60


def build_oracle_seed_messages(
    scenario: FactorScenario,
    *,
    namespace: str,
    count: int,
) -> list[dict[str, str]]:
    """Build messages asking an oracle LLM for raw DSL seed expressions."""

    benchmark = get_benchmark(scenario.benchmark)
    fields = tuple(field for field in benchmark.fields if field.startswith(f"{namespace}."))
    operators = ", ".join(sorted(OPERATORS))
    field_text = ", ".join(fields)
    examples = "\n".join(f"- {expr}" for expr in benchmark.examples if f"{namespace}." in expr)

    system = (
        "You are an expert quantitative researcher constructing initial alpha-factor seeds. "
        "Return only DSL expressions that obey the provided grammar. "
        "Each expression must use exactly the requested namespace fields and no Python code."
    )
    user = f"""
Scenario:
- name: {scenario.name}
- benchmark: {scenario.benchmark}
- market: {scenario.market}
- horizon: {scenario.horizon}
- objective: {scenario.objective}
- constraints: {", ".join(scenario.constraints) if scenario.constraints else "none"}

Allowed namespace: {namespace}
Allowed fields: {field_text}
Allowed operators: {operators}

Examples:
{examples or "- none"}

Generate {int(count)} diverse candidate factor expressions.
Output format:
<expr>one_expression_here</expr>
<expr>another_expression_here</expr>

Do not include explanations, markdown fences, imports, assignments, or prose.
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_seed_expressions(text: str) -> tuple[str, ...]:
    """Extract candidate expressions from oracle LLM output."""

    tagged = tuple(expr.strip() for expr in _EXPR_RE.findall(text) if expr.strip())
    if tagged:
        return tagged

    lines = []
    for raw in str(text).splitlines():
        line = raw.strip().lstrip("-0123456789. ").strip()
        if line:
            lines.append(line)
    return tuple(lines)


def generate_oracle_seed_candidates(
    scenario: FactorScenario,
    *,
    namespace: str,
    count: int = 24,
    config: OpenRouterConfig | None = None,
    api_key: str | None = None,
) -> tuple[str, ...]:
    """Generate raw seed candidates with OpenRouter and return valid DSL expressions."""

    cfg = config or OpenRouterConfig()
    data = call_openrouter_chat(
        model=cfg.model,
        messages=build_oracle_seed_messages(scenario, namespace=namespace, count=count),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout_seconds=cfg.timeout_seconds,
        api_key=api_key,
    )
    content = openrouter_content(data)
    return _valid_expressions_for_namespace(extract_seed_expressions(content), namespace)


def generate_oracle_seed_candidates_from_file(path: Path, *, namespace: str) -> tuple[str, ...]:
    """Load oracle output from a file and extract valid expressions."""

    return _valid_expressions_for_namespace(extract_seed_expressions(path.read_text()), namespace)


def _valid_expressions_for_namespace(expressions: tuple[str, ...], namespace: str) -> tuple[str, ...]:
    valid = []
    for expr in expressions:
        try:
            validate_expr(expr)
        except Exception:
            continue
        if _expr_uses_only_namespace(expr, namespace):
            valid.append(expr)
    return tuple(valid)


def _expr_uses_only_namespace(expr: str, namespace: str) -> bool:
    fields = _collect_fields(parse_expr(expr))
    return bool(fields) and all(field.namespace == namespace for field in fields)


def _collect_fields(node: ExprNode) -> tuple[FieldNode, ...]:
    if isinstance(node, FieldNode):
        return (node,)
    if isinstance(node, CallNode):
        fields: list[FieldNode] = []
        for arg in node.args:
            fields.extend(_collect_fields(arg))
        return tuple(fields)
    if isinstance(node, NumberNode):
        return ()
    raise TypeError(f"unsupported node type: {type(node).__name__}")
