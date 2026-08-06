from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from examples.dsl_smoke_test import (
    DEFAULT_CRYPTO_PANEL,
    fetch_wrds_crsp_dsf_v2_sample,
    normalize_wrds_password,
    run_crypto_smoke,
    wrds_env_status,
)


def test_crypto_smoke_runs_on_saved_project_data():
    if not DEFAULT_CRYPTO_PANEL.exists():
        pytest.skip("saved crypto project panel is not available")

    result = run_crypto_smoke(DEFAULT_CRYPTO_PANEL, "BTC-USD")

    assert result["crypto_rows"] >= 25
    assert result["crypto_value"] == pytest.approx(result["crypto_value"])


def test_wrds_env_status_does_not_require_connection():
    status = wrds_env_status()

    assert set(status) == {"wrds_userid_present"}
    assert isinstance(status["wrds_userid_present"], bool)


def test_normalize_wrds_password_accepts_raw_or_pgpass_line():
    assert normalize_wrds_password("secret") == "secret"
    assert normalize_wrds_password("wrds-pgdata.wharton.upenn.edu:9737:wrds:user:secret") == "secret"


def test_wrds_sample_fetch_requires_userid(monkeypatch):
    monkeypatch.delenv("WRDS_USERID", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.delenv("WRDS_PGPASS", raising=False)

    with patch("examples.dsl_smoke_test.load_dotenv"):
        with pytest.raises(RuntimeError, match="WRDS_USERID"):
            fetch_wrds_crsp_dsf_v2_sample()


def test_wrds_sample_fetch_requires_noninteractive_credentials(monkeypatch):
    monkeypatch.setenv("WRDS_USERID", "test_user")
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.delenv("WRDS_PGPASS", raising=False)

    with patch("examples.dsl_smoke_test.load_dotenv"):
        with pytest.raises(RuntimeError, match="non-interactive credentials"):
            fetch_wrds_crsp_dsf_v2_sample()


def test_wrds_sample_fetch_uses_parameterized_query(monkeypatch):
    monkeypatch.setenv("WRDS_USERID", "test_user")
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.setenv("WRDS_PGPASS", "test_password")
    sample = pd.DataFrame(
        {
            "permno": [14593],
            "dlycaldt": pd.to_datetime(["2023-01-03"]),
            "dlyopen": [100.0],
            "dlyhigh": [101.0],
            "dlylow": [99.0],
            "dlyclose": [100.5],
            "dlyvol": [1_000_000.0],
            "dlyret": [0.01],
            "dlycap": [2_000_000.0],
        }
    )
    connection = MagicMock()
    connection.raw_sql.return_value = sample

    with patch("examples.dsl_smoke_test.load_dotenv"):
        with patch.dict("sys.modules", {"wrds": MagicMock(Connection=MagicMock(return_value=connection))}):
            out = fetch_wrds_crsp_dsf_v2_sample(
                permno=14593,
                start_date="2023-01-01",
                end_date="2023-03-31",
            )

    raw_sql = connection.raw_sql
    query = raw_sql.call_args.args[0]
    kwargs = raw_sql.call_args.kwargs

    assert out.equals(sample)
    assert "from crsp.dsf_v2" in query
    assert "%(permno)s" in query
    assert kwargs["params"] == {
        "permno": 14593,
        "start_date": "2023-01-01",
        "end_date": "2023-03-31",
    }
    assert kwargs["date_cols"] == ["dlycaldt"]
