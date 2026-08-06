import pytest

from src.dsl.fields import get_field, list_fields


def test_crsp_fields_are_registered():
    fields = list_fields("crsp")
    assert "crsp.dlyret" in fields
    assert "crsp.dlycap" in fields
    assert "crsp.dlyclose" in fields


def test_taq_and_crypto_fields_are_registered():
    assert "taq.spread" in list_fields("taq")
    assert "taq.imbalance" in list_fields("taq")
    assert "crypto.returns" in list_fields("crypto")
    assert "crypto.volume" in list_fields("crypto")


def test_get_field_returns_metadata():
    spec = get_field("crsp.dlyret")
    assert spec.namespace == "crsp"
    assert spec.name == "dlyret"
    assert spec.kind == "return"
    assert spec.source_column == "dlyret"


def test_unknown_field_fails():
    with pytest.raises(ValueError, match="unknown field"):
        get_field("crsp.not_a_field")

