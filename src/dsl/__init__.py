"""DSL namespace, field, and operator primitives."""

from src.dsl.benchmarks import BenchmarkSpec, get_benchmark, list_benchmark_examples, list_benchmarks
from src.dsl.context import FieldWindow, PointInTimeContext
from src.dsl.evaluator import EvaluationError, evaluate_expr, evaluate_node
from src.dsl.fields import FieldSpec, get_field, list_fields
from src.dsl.namespaces import NamespaceSpec, get_namespace, list_namespaces
from src.dsl.parser import CallNode, FieldNode, NumberNode, ParseError, parse_expr
from src.dsl.validator import ValidationConfig, ValidationError, validate_expr

__all__ = [
    "CallNode",
    "BenchmarkSpec",
    "FieldSpec",
    "FieldNode",
    "FieldWindow",
    "EvaluationError",
    "NamespaceSpec",
    "NumberNode",
    "ParseError",
    "PointInTimeContext",
    "ValidationConfig",
    "ValidationError",
    "evaluate_expr",
    "evaluate_node",
    "get_field",
    "get_benchmark",
    "get_namespace",
    "list_benchmark_examples",
    "list_benchmarks",
    "list_fields",
    "list_namespaces",
    "parse_expr",
    "validate_expr",
]
