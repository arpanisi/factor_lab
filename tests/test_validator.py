import pytest

from src.dsl.parser import CallNode
from src.dsl.validator import ValidationConfig, ValidationError, validate_expr


def test_validate_good_expression():
    node = validate_expr("div(ts_mean(crsp.dlyret(20)), ts_std(crsp.dlyret(20)))")
    assert isinstance(node, CallNode)
    assert node.value_type == "scalar"


def test_validate_crypto_expression():
    node = validate_expr("div(ts_mean(crypto.returns(24)), ts_std(crypto.returns(24)))")
    assert node.value_type == "scalar"


def test_validate_taq_expression():
    node = validate_expr("div(ts_mean(taq.spread(30)), ts_std(taq.midret(30)))")
    assert node.value_type == "scalar"


def test_unknown_field_fails_validation():
    with pytest.raises(ValidationError, match="unknown field"):
        validate_expr("ts_mean(crsp.nope(20))")


def test_window_below_min_fails():
    with pytest.raises(ValidationError, match="window must be"):
        validate_expr("ts_mean(crsp.dlyret(0))")


def test_window_above_max_fails():
    with pytest.raises(ValidationError, match="exceeds max_window"):
        validate_expr("ts_mean(crsp.dlyret(9999))", ValidationConfig(max_window=252))


def test_parse_errors_are_validation_errors():
    with pytest.raises(ValidationError, match="unknown operator"):
        validate_expr("future_return(crsp.dlyret(20))")
