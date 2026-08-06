from unittest.mock import patch

import pandas as pd
import pytest

from examples.build_seed_bank import (
    build_crypto_cross_sectional_seed_bank,
    build_wrds_crsp_single_asset_seed_bank,
    crypto_frames_from_panel,
)
from examples.dsl_smoke_test import DEFAULT_CRYPTO_PANEL


def test_crypto_frames_from_panel_converts_saved_shape():
    if not DEFAULT_CRYPTO_PANEL.exists():
        pytest.skip("saved crypto project panel is not available")

    panel = pd.read_pickle(DEFAULT_CRYPTO_PANEL)
    frames = crypto_frames_from_panel(panel, tickers=("BTC-USD", "ETH-USD"))

    assert set(frames) == {"BTC-USD", "ETH-USD"}
    assert "returns" in frames["BTC-USD"].columns


def test_build_crypto_cross_sectional_seed_bank_from_saved_panel():
    if not DEFAULT_CRYPTO_PANEL.exists():
        pytest.skip("saved crypto project panel is not available")

    payload = build_crypto_cross_sectional_seed_bank(
        panel_path=DEFAULT_CRYPTO_PANEL,
        tickers=("BTC-USD", "ETH-USD", "XRP-USD"),
        top_k=2,
        quality_threshold=-1.0,
    )

    result = payload["result"]
    assert payload["asset_count"] == 3
    assert len(result.seeds) <= 2
    assert len(result.tasks) == len(result.seeds)


def test_build_wrds_crsp_single_asset_seed_bank_with_mocked_fetch():
    raw = pd.DataFrame(
        {
            "dlycaldt": pd.date_range("2023-01-01", periods=40, freq="D"),
            "dlyopen": range(40),
            "dlyhigh": range(1, 41),
            "dlylow": range(40),
            "dlyclose": range(10, 50),
            "dlyvol": [1000] * 40,
            "dlyret": [0.01] * 40,
            "dlycap": [1_000_000] * 40,
        }
    )

    with patch("examples.build_seed_bank.fetch_wrds_crsp_dsf_v2_sample", return_value=raw):
        payload = build_wrds_crsp_single_asset_seed_bank(top_k=2, quality_threshold=-1.0)

    result = payload["result"]
    assert payload["row_count"] == 40
    assert len(result.seeds) <= 2
    assert len(result.tasks) == len(result.seeds)
