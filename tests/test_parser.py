import pytest

from src.dsl.parser import CallNode, FieldNode, NumberNode, ParseError, parse_expr


def test_parse_namespaced_field_call():
    node = parse_expr("crsp.dlyret(20)")
    assert isinstance(node, FieldNode)
    assert node.namespace == "crsp"
    assert node.field == "dlyret"
    assert node.window == 20
    assert node.value_type == "array"


def test_parse_nested_expression_types():
    node = parse_expr("div(ts_mean(crsp.dlyret(20)), ts_std(crsp.dlyret(20)))")
    assert isinstance(node, CallNode)
    assert node.name == "div"
    assert node.value_type == "scalar"
    assert all(arg.value_type == "scalar" for arg in node.args)


def test_parse_infix_expression_as_call():
    node = parse_expr("ts_mean(crsp.dlyret(20)) / ts_std(crsp.dlyret(20))")
    assert isinstance(node, CallNode)
    assert node.name == "div"
    assert node.value_type == "scalar"


def test_parse_numeric_constant():
    node = parse_expr("-2.5")
    assert isinstance(node, NumberNode)
    assert node.value == -2.5
    assert node.value_type == "scalar"


def test_unknown_operator_fails():
    with pytest.raises(ParseError, match="unknown operator"):
        parse_expr("future_return(crsp.dlyret(20))")


def test_field_window_must_be_integer_literal():
    with pytest.raises(ParseError, match="integer constant"):
        parse_expr("crsp.dlyret(x)")


def test_wrong_operator_type_fails():
    with pytest.raises(ParseError, match="expects arg types"):
        parse_expr("ts_mean(1.0)")


def test_keyword_arguments_fail():
    with pytest.raises(ParseError, match="keyword arguments"):
        parse_expr("ts_mean(crsp.dlyret(window=20))")

