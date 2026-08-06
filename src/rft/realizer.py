"""Factor realization from miner output."""

from __future__ import annotations

from dataclasses import dataclass

from src.dsl.parser import ExprNode
from src.dsl.validator import validate_expr
from src.seeds.canonical import canonical_signature
from src.seeds.oracle import extract_seed_expressions


@dataclass(frozen=True)
class RealizedFactor:
    """Validated candidate factor."""

    expr: str
    node: ExprNode
    signature: str
    valid: bool
    reason: str = ""


def extract_factor_expressions(completion: str) -> tuple[str, ...]:
    """Extract candidate expressions from miner completion text."""

    return extract_seed_expressions(completion)


def realize_factor(expr: str) -> RealizedFactor:
    """Validate a generated DSL expression and return realization metadata."""

    expr = str(expr).strip()
    try:
        node = validate_expr(expr)
        signature = canonical_signature(node)
    except Exception as exc:
        return RealizedFactor(expr=expr, node=None, signature="", valid=False, reason=str(exc))  # type: ignore[arg-type]
    return RealizedFactor(expr=expr, node=node, signature=signature, valid=True)
