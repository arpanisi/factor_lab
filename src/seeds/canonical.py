"""Canonical signatures for DSL expressions."""

from __future__ import annotations

import hashlib

from src.dsl.parser import CallNode, ExprNode, FieldNode, NumberNode, parse_expr


_COMMUTATIVE_OPS = {"add", "mul", "corr", "cov"}


def canonical_form(expr_or_node: str | ExprNode) -> str:
    """Return a normalized expression form for deduplication."""

    node = parse_expr(expr_or_node) if isinstance(expr_or_node, str) else expr_or_node
    return _canonical_node(node)


def canonical_signature(expr_or_node: str | ExprNode) -> str:
    """Return a stable hash for a normalized expression."""

    form = canonical_form(expr_or_node)
    return hashlib.sha256(form.encode("utf-8")).hexdigest()


def _canonical_node(node: ExprNode) -> str:
    if isinstance(node, NumberNode):
        return f"num:{node.value:g}"

    if isinstance(node, FieldNode):
        return f"field:{node.namespace.lower()}.{node.field.lower()}:{node.window}"

    if isinstance(node, CallNode):
        args = tuple(_canonical_node(arg) for arg in node.args)
        name = node.name.lower()
        if name in _COMMUTATIVE_OPS:
            args = tuple(sorted(args))
        return f"call:{name}({','.join(args)})"

    raise TypeError(f"unsupported node type: {type(node).__name__}")
