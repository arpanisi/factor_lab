"""Parser for the Factor Lab DSL."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal, Union


ValueType = Literal["array", "scalar"]


@dataclass(frozen=True)
class NumberNode:
    value: float
    value_type: ValueType = "scalar"


@dataclass(frozen=True)
class FieldNode:
    namespace: str
    field: str
    window: int
    value_type: ValueType = "array"

    @property
    def full_name(self) -> str:
        return f"{self.namespace}.{self.field}"


@dataclass(frozen=True)
class CallNode:
    name: str
    args: tuple["ExprNode", ...]
    value_type: ValueType


ExprNode = Union[NumberNode, FieldNode, CallNode]


class ParseError(ValueError):
    """Raised when DSL text cannot be parsed into an expression tree."""


_ARRAY_TO_SCALAR_OPS = {
    "ts_mean",
    "ts_std",
    "ts_sum",
    "ts_min",
    "ts_max",
    "first",
    "last",
}
_ARRAY_TO_ARRAY_OPS = {"diff", "log", "normalize", "ema", "zscore", "rank"}
_BINARY_ARRAY_TO_SCALAR_OPS = {"corr", "cov"}
_BINARY_SAME_TYPE_OPS = {"add", "sub", "mul", "div"}
_UNARY_SAME_TYPE_OPS = {"neg", "abs", "abs_s", "tanh", "tanh_s", "sign", "sign_s"}


def parse_expr(expr: str) -> ExprNode:
    """Parse a DSL expression string into a typed expression tree."""

    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except SyntaxError as exc:
        raise ParseError(f"DSL syntax error: {exc}") from exc

    return _AstParser().parse(tree.body)


class _AstParser:
    def parse(self, node: ast.AST) -> ExprNode:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return NumberNode(float(node.value))

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            child = self.parse(node.operand)
            if not isinstance(child, NumberNode):
                raise ParseError("unary minus is only allowed for numeric constants")
            return NumberNode(-child.value)

        if isinstance(node, ast.BinOp):
            return self._parse_binop(node)

        if isinstance(node, ast.Call):
            return self._parse_call(node)

        raise ParseError(f"unsupported DSL syntax: {ast.dump(node, include_attributes=False)}")

    def _parse_binop(self, node: ast.BinOp) -> CallNode:
        op_name_by_type = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
        }
        op_name = op_name_by_type.get(type(node.op))
        if op_name is None:
            raise ParseError("only +, -, *, and / infix operators are supported")

        left = self.parse(node.left)
        right = self.parse(node.right)
        value_type = _binary_output_type(op_name, left.value_type, right.value_type)
        return CallNode(op_name, (left, right), value_type)

    def _parse_call(self, node: ast.Call) -> ExprNode:
        if node.keywords:
            raise ParseError("keyword arguments are not allowed in DSL calls")

        if isinstance(node.func, ast.Attribute):
            return self._parse_field_call(node)

        if not isinstance(node.func, ast.Name):
            raise ParseError("only simple operator calls and namespace.field calls are supported")

        name = node.func.id
        args = tuple(self.parse(arg) for arg in node.args)
        value_type = _operator_output_type(name, tuple(arg.value_type for arg in args))
        return CallNode(name, args, value_type)

    def _parse_field_call(self, node: ast.Call) -> FieldNode:
        if not isinstance(node.func.value, ast.Name):
            raise ParseError("field calls must use namespace.field(window) syntax")
        if len(node.args) != 1:
            raise ParseError("field calls require exactly one window argument")
        if node.keywords:
            raise ParseError("keyword arguments are not allowed in field calls")

        window_node = node.args[0]
        if not isinstance(window_node, ast.Constant) or not isinstance(window_node.value, int):
            raise ParseError("field window must be an integer constant")

        return FieldNode(
            namespace=node.func.value.id,
            field=node.func.attr,
            window=int(window_node.value),
        )


def _operator_output_type(name: str, arg_types: tuple[ValueType, ...]) -> ValueType:
    if name in _ARRAY_TO_SCALAR_OPS:
        _expect_arity(name, arg_types, 1)
        _expect_types(name, arg_types, ("array",))
        return "scalar"

    if name in _ARRAY_TO_ARRAY_OPS:
        _expect_arity(name, arg_types, 1)
        _expect_types(name, arg_types, ("array",))
        return "array"

    if name in _BINARY_ARRAY_TO_SCALAR_OPS:
        _expect_arity(name, arg_types, 2)
        _expect_types(name, arg_types, ("array", "array"))
        return "scalar"

    if name in _BINARY_SAME_TYPE_OPS:
        _expect_arity(name, arg_types, 2)
        return _binary_output_type(name, arg_types[0], arg_types[1])

    if name in _UNARY_SAME_TYPE_OPS:
        _expect_arity(name, arg_types, 1)
        return arg_types[0]

    raise ParseError(f"unknown operator '{name}'")


def _binary_output_type(name: str, left_type: ValueType, right_type: ValueType) -> ValueType:
    if left_type == right_type:
        return left_type
    if left_type == "array" and right_type == "scalar":
        return "array"
    if left_type == "scalar" and right_type == "array":
        return "array"
    raise ParseError(f"unsupported operand types for {name}: {left_type}, {right_type}")


def _expect_arity(name: str, arg_types: tuple[ValueType, ...], expected: int) -> None:
    if len(arg_types) != expected:
        raise ParseError(f"operator '{name}' expects {expected} args, got {len(arg_types)}")


def _expect_types(name: str, arg_types: tuple[ValueType, ...], expected: tuple[ValueType, ...]) -> None:
    if arg_types != expected:
        raise ParseError(f"operator '{name}' expects arg types {expected}, got {arg_types}")
