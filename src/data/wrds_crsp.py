"""Build WRDS/CRSP daily datasets for Factor Lab."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adapters import adapt_crsp_dsf_v2

CRSP_DSF_V2_COLUMNS = [
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


def normalize_wrds_password(value: str | None) -> str:
    """Return password from a raw password or pgpass-style entry."""

    if value is None:
        return ""
    stripped = value.strip()
    parts = stripped.split(":", 4)
    if len(parts) == 5 and parts[0] and parts[1] and parts[2] and parts[3]:
        return parts[4]
    return stripped


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE lines into ``os.environ`` if not already set."""

    candidates = [path] if path is not None else [REPO_ROOT / ".env", REPO_ROOT / "factor_lab" / ".env"]
    env_path = next((candidate for candidate in candidates if candidate and candidate.exists()), None)
    if env_path is None:
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def connect_wrds():
    """Create a non-interactive WRDS connection using local env/.env values."""

    load_dotenv()
    userid = os.getenv("WRDS_USERID")
    if not userid:
        raise RuntimeError("WRDS_USERID is not set")
    password_source = os.getenv("WRDS_PASSWORD") or os.getenv("PGPASSWORD") or os.getenv("WRDS_PGPASS")
    password = normalize_wrds_password(password_source)
    _configure_pgpass(userid, password_source)
    if password and not os.getenv("PGPASSWORD"):
        os.environ["PGPASSWORD"] = password

    try:
        import wrds
    except ImportError as exc:
        raise RuntimeError("Python package 'wrds' is required") from exc

    kwargs: dict[str, Any] = {"wrds_username": userid}
    if password:
        kwargs["wrds_password"] = password
    return wrds.Connection(**kwargs)


def _configure_pgpass(userid: str, password_source: str | None) -> None:
    """Configure a temporary pgpass file from WRDS_PGPASS when provided."""

    if os.getenv("PGPASSFILE") or not password_source:
        return
    stripped = password_source.strip()
    if stripped.count(":") < 4:
        return
    parts = stripped.split(":", 4)
    if len(parts) != 5:
        return
    if not parts[3]:
        parts[3] = userid
    pgpass_line = ":".join(parts)
    path = Path(tempfile.gettempdir()) / "factor_lab_wrds.pgpass"
    path.write_text(pgpass_line + "\n")
    path.chmod(0o600)
    os.environ["PGPASSFILE"] = str(path)


def build_crsp_dsf_v2_query(
    *,
    start_date: str,
    end_date: str,
    universe: str = "sp500",
    max_assets: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a parameterized WRDS query for CRSP daily data."""

    columns = ", ".join(f"d.{col}" for col in CRSP_DSF_V2_COLUMNS)
    filters = [
        "d.dlycaldt between %(start_date)s and %(end_date)s",
        "d.dlyclose is not null",
        "d.dlyret is not null",
    ]
    join = ""
    if universe == "sp500":
        join = """
        join crsp.dsp500list s
          on d.permno = s.permno
         and d.dlycaldt >= s.start
         and d.dlycaldt <= coalesce(s.ending, '9999-12-31'::date)
        """
    elif universe != "all":
        raise ValueError("universe must be 'sp500' or 'all'")

    query = f"""
        select {columns}
        from crsp.dsf_v2 d
        {join}
        where {' and '.join(filters)}
    """
    params: dict[str, Any] = {"start_date": start_date, "end_date": end_date}

    if max_assets is not None and int(max_assets) > 0:
        query = f"""
            with base as (
                {query}
            ),
            ranked_assets as (
                select permno
                from base
                group by permno
                order by avg(abs(dlycap)) desc nulls last
                limit %(max_assets)s
            )
            select b.*
            from base b
            join ranked_assets r on b.permno = r.permno
            order by b.dlycaldt, b.permno
        """
        params["max_assets"] = int(max_assets)
    else:
        query = f"{query} order by d.dlycaldt, d.permno"

    return query, params


def fetch_crsp_dsf_v2_panel(
    *,
    start_date: str,
    end_date: str,
    universe: str = "sp500",
    max_assets: int | None = None,
) -> pd.DataFrame:
    """Fetch CRSP daily rows from WRDS."""

    query, params = build_crsp_dsf_v2_query(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        max_assets=max_assets,
    )
    db = connect_wrds()
    try:
        return db.raw_sql(query, params=params, date_cols=["dlycaldt"])
    finally:
        db.close()


def clean_crsp_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean WRDS CRSP rows into a long DSL-ready panel."""

    if raw.empty:
        raise ValueError("WRDS query returned zero CRSP rows")
    required = set(CRSP_DSF_V2_COLUMNS)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"CRSP result missing columns: {missing}")

    out = raw.copy()
    out["dlycaldt"] = pd.to_datetime(out["dlycaldt"])
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    value_cols = [col for col in CRSP_DSF_V2_COLUMNS if col not in {"permno", "dlycaldt"}]
    out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["permno", "dlycaldt", "dlyclose", "dlyret"])
    out = out.sort_values(["dlycaldt", "permno"]).drop_duplicates(["dlycaldt", "permno"], keep="last")
    return out


def crsp_frames_from_long_panel(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert long CRSP rows into per-PERMNO frames for DSL evaluation."""

    frames: dict[str, pd.DataFrame] = {}
    for permno, group in panel.groupby("permno", sort=True):
        frame = adapt_crsp_dsf_v2(group, date_col="dlycaldt")
        frames[str(int(permno))] = frame
    return frames


def save_crsp_outputs(panel: pd.DataFrame, *, parquet_path: Path, frames_pickle_path: Path) -> dict[str, Any]:
    """Save long parquet plus per-asset frame pickle."""

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frames_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(parquet_path, index=False)
    frames = crsp_frames_from_long_panel(panel)
    pd.to_pickle(frames, frames_pickle_path)
    return {
        "rows": int(panel.shape[0]),
        "assets": int(len(frames)),
        "start": str(panel["dlycaldt"].min().date()),
        "end": str(panel["dlycaldt"].max().date()),
        "parquet": str(parquet_path),
        "frames_pickle": str(frames_pickle_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and save a WRDS/CRSP daily panel.")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--universe", choices=("sp500", "all"), default="sp500")
    parser.add_argument("--max-assets", type=int, default=500)
    parser.add_argument("--output-parquet", type=Path, default=Path("data/wrds_crsp_daily_panel.parquet"))
    parser.add_argument("--output-frames", type=Path, default=Path("data/wrds_crsp_daily_frames.pkl"))
    args = parser.parse_args(argv)

    raw = fetch_crsp_dsf_v2_panel(
        start_date=args.start_date,
        end_date=args.end_date,
        universe=args.universe,
        max_assets=args.max_assets,
    )
    panel = clean_crsp_panel(raw)
    summary = save_crsp_outputs(panel, parquet_path=args.output_parquet, frames_pickle_path=args.output_frames)
    print(
        "saved WRDS CRSP panel: "
        f"rows={summary['rows']} assets={summary['assets']} "
        f"range={summary['start']}..{summary['end']} "
        f"parquet={summary['parquet']} frames={summary['frames_pickle']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
