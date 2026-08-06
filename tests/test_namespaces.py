import pytest

from src.dsl.namespaces import get_namespace, list_namespaces


def test_initial_namespaces_are_registered():
    assert list_namespaces() == ("crsp", "crypto", "taq")


def test_get_namespace_returns_metadata():
    spec = get_namespace("crsp")
    assert spec.name == "crsp"
    assert spec.time_scale == "daily"


def test_unknown_namespace_fails():
    with pytest.raises(ValueError, match="unknown namespace"):
        get_namespace("fundamental")

