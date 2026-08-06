from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.wrds_crsp import (
    build_crsp_dsf_v2_query,
    clean_crsp_panel,
    crsp_frames_from_long_panel,
    fetch_crsp_dsf_v2_panel,
    save_crsp_outputs,
    _configure_pgpass,
)


def _raw():
    return pd.DataFrame(
        {
            "permno": [1, 1, 2],
            "dlycaldt": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02"]),
            "dlyopen": [10, 11, 20],
            "dlyhigh": [12, 13, 22],
            "dlylow": [9, 10, 19],
            "dlyclose": [11, 12, 21],
            "dlyvol": [1000, 1100, 2000],
            "dlyret": [0.01, 0.02, -0.01],
            "dlycap": [100000, 110000, 200000],
        }
    )


def test_build_crsp_dsf_v2_query_uses_sp500_membership_and_limit():
    query, params = build_crsp_dsf_v2_query(
        start_date="2019-01-01",
        end_date="2024-12-31",
        universe="sp500",
        max_assets=100,
    )

    assert "from crsp.dsf_v2 d" in query
    assert "join crsp.dsp500list s" in query
    assert "d.dlycaldt >= s.start" in query
    assert "limit %(max_assets)s" in query
    assert params == {"start_date": "2019-01-01", "end_date": "2024-12-31", "max_assets": 100}


def test_fetch_crsp_dsf_v2_panel_uses_parameterized_query():
    connection = MagicMock()
    connection.raw_sql.return_value = _raw()

    with patch("src.data.wrds_crsp.connect_wrds", return_value=connection):
        out = fetch_crsp_dsf_v2_panel(
            start_date="2019-01-01",
            end_date="2024-12-31",
            universe="sp500",
            max_assets=50,
        )

    assert out.equals(_raw())
    kwargs = connection.raw_sql.call_args.kwargs
    assert kwargs["params"]["max_assets"] == 50
    assert kwargs["date_cols"] == ["dlycaldt"]
    connection.close.assert_called_once()


def test_clean_and_save_crsp_panel(tmp_path):
    panel = clean_crsp_panel(_raw())
    frames = crsp_frames_from_long_panel(panel)

    assert list(panel.columns) == [
        "permno",
        "dlycaldt",
        "dlyopen",
        "dlyhigh",
        "dlylow",
        "dlyclose",
        "dlyvol",
        "dlyret",
        "dlycap",
    ]
    assert set(frames) == {"1", "2"}
    assert "dlyret" in frames["1"].columns

    with patch.object(pd.DataFrame, "to_parquet") as to_parquet:
        summary = save_crsp_outputs(
            panel,
            parquet_path=tmp_path / "crsp.parquet",
            frames_pickle_path=tmp_path / "frames.pkl",
        )
    assert summary["rows"] == 3
    assert summary["assets"] == 2
    to_parquet.assert_called_once()
    assert (tmp_path / "frames.pkl").exists()


def test_configure_pgpass_writes_temp_file(monkeypatch):
    monkeypatch.delenv("PGPASSFILE", raising=False)
    monkeypatch.setenv("TMPDIR", "/tmp")

    _configure_pgpass("user", "host:9737:wrds:user:secret")

    import os
    from pathlib import Path

    path = Path(os.environ["PGPASSFILE"])
    assert path.exists()
    assert path.read_text().strip().endswith(":secret")
