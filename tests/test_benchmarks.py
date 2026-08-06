import pytest

from src.dsl import get_benchmark, list_benchmark_examples, list_benchmarks, validate_expr


def test_three_benchmark_surfaces_are_registered():
    assert list_benchmarks() == (
        "daily_cross_sectional_rankic",
        "intraday_microstructure_direction",
        "single_asset_direction",
    )


def test_benchmark_fields_are_explicit():
    spec = get_benchmark("daily_cross_sectional_rankic")
    assert "crsp.dlyret" in spec.fields
    assert "crypto.returns" in spec.fields
    assert "taq.spread" not in spec.fields


def test_every_benchmark_example_validates():
    for benchmark in list_benchmarks():
        for expr in list_benchmark_examples(benchmark):
            node = validate_expr(expr)
            assert node.value_type in {"array", "scalar"}


def test_unknown_benchmark_fails():
    with pytest.raises(ValueError, match="unknown benchmark"):
        get_benchmark("not_a_benchmark")
