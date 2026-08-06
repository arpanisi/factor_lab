"""Evaluate parsed Factor Lab DSL expressions."""

from __future__ import annotations

import numpy as np

from src.dsl.context import PointInTimeContext
from src.dsl.operators import get_operator
from src.dsl.parser import CallNode, ExprNode, FieldNode, NumberNode
from src.dsl.validator import ValidationConfig, validate_expr


class EvaluationError(ValueError):
    """Raised when an expression cannot be evaluated."""


def evaluate_node(node: ExprNode, context: PointInTimeContext):
    """Evaluate a parsed expression node against a point-in-time context."""

    if isinstance(node, NumberNode):
        return float(node.value)

    if isinstance(node, FieldNode):
        return context.window(node.full_name, node.window).values

    if isinstance(node, CallNode):
        fn = get_operator(node.name)
        args = [evaluate_node(arg, context) for arg in node.args]
        try:
            value = fn(*args)
        except Exception as exc:
            raise EvaluationError(f"operator '{node.name}' failed: {exc}") from exc
        return value

    raise EvaluationError(f"unsupported node type: {type(node).__name__}")


def evaluate_expr(expr: str, context: PointInTimeContext, config: ValidationConfig | None = None) -> float:
    """Validate and evaluate a DSL expression into one scalar factor value."""

    node = validate_expr(expr, config=config)
    value = evaluate_node(node, context)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        value = value[-1]

    try:
        return float(value)
    except Exception as exc:
        raise EvaluationError(f"expression did not produce a scalar-compatible value: {value}") from exc

