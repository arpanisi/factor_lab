"""Validation for parsed Factor Lab DSL expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.dsl.fields import get_field
from src.dsl.operators import OPERATORS
from src.dsl.parser import CallNode, ExprNode, FieldNode, NumberNode, ParseError, parse_expr


@dataclass(frozen=True)
class ValidationConfig:
    """Limits for accepted DSL expressions."""

    min_window: int = 1
    max_window: int = 2520
    max_depth: int = 16


class ValidationError(ValueError):
    """Raised when a parsed DSL expression violates registry or safety rules."""


def validate_expr(expr: str, config: Optional[ValidationConfig] = None) -> ExprNode:
    """Parse and validate a DSL expression, returning the typed expression tree."""

    cfg = config or ValidationConfig()
    try:
        node = parse_expr(expr)
    except ParseError as exc:
        raise ValidationError(str(exc)) from exc

    _validate_node(node, cfg, depth=0)
    return node


def _validate_node(node: ExprNode, cfg: ValidationConfig, depth: int) -> None:
    if depth > cfg.max_depth:
        raise ValidationError(f"expression depth exceeds max_depth={cfg.max_depth}")

    if isinstance(node, NumberNode):
        return

    if isinstance(node, FieldNode):
        try:
            get_field(node.full_name)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if node.window < cfg.min_window:
            raise ValidationError(f"field '{node.full_name}' window must be >= {cfg.min_window}")
        if node.window > cfg.max_window:
            raise ValidationError(f"field '{node.full_name}' window exceeds max_window={cfg.max_window}")
        return

    if isinstance(node, CallNode):
        if node.name not in OPERATORS:
            raise ValidationError(f"unknown operator '{node.name}'")
        for arg in node.args:
            _validate_node(arg, cfg, depth + 1)
        return

    raise ValidationError(f"unknown node type: {type(node).__name__}")
