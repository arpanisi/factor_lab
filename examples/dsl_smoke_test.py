"""End-to-end smoke checks for Factor Lab DSL expressions.

The default path uses local artifacts already present in the repo. WRDS access
is optional and only checked through environment variables; credentials are not
printed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.data import adapt_crypto_ohlcv, adapt_crsp_dsf_v2
from src.dsl import PointInTimeContext, evaluate_expr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRYPTO_PANEL = REPO_ROOT / "data" / "crypto" / "crypto_panel_clean.pkl"
DEFAULT_CRSP_PARQUET: Path | None = None

CRYPTO_EXPR = "div(ts_mean(crypto.returns(24)), ts_std(crypto.returns(24)))"
CRSP_EXPR = "div(ts_mean(crsp.dlyret(20)), ts_std(crsp.dlyret(20)))"
WRDS_CRSP_COLUMNS = [
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


def crypto_asset_frame_from_panel(panel: dict, ticker: str) -> pd.DataFrame:
    """Build one asset OHLCV frame from the saved crypto panel dictionary."""

    required = ["open", "high", "low", "close", "volume"]
    missing = [key for key in required if key not in panel]
    if missing:
        raise ValueError(f"crypto panel missing keys: {missing}")

    frame = pd.DataFrame({key: panel[key][ticker] for key in required})
    if "returns" in panel:
        frame["returns"] = panel["returns"][ticker]
    return frame


def evaluate_last_available(
    namespace: str,
    frame: pd.DataFrame,
    expr: str,
    *,
    min_history: int,
) -> float:
    """Evaluate one expression at the last timestamp with enough history."""

    clean = frame.dropna()
    if clean.shape[0] < min_history:
        raise ValueError(f"not enough observations: {clean.shape[0]} < {min_history}")

    timestamp = clean.index[-1]
    ctx = PointInTimeContext({namespace: clean}, timestamp)
    value = evaluate_expr(expr, ctx)
    if not pd.notna(value):
        raise ValueError(f"expression returned non-finite value at {timestamp}")
    return float(value)


def run_crypto_smoke(panel_path: Path = DEFAULT_CRYPTO_PANEL, ticker: str = "BTC-USD") -> dict[str, float]:
    """Run the crypto namespace smoke test on the saved project panel."""

    panel = pd.read_pickle(panel_path)
    frame = crypto_asset_frame_from_panel(panel, ticker)
    adapted = adapt_crypto_ohlcv(frame)
    value = evaluate_last_available("crypto", adapted, CRYPTO_EXPR, min_history=25)
    return {"crypto_value": value, "crypto_rows": float(adapted.shape[0])}


def run_crsp_local_smoke(parquet_path: Path) -> dict[str, float]:
    """Run the CRSP namespace smoke test if the local parquet can be read."""

    raw = pd.read_parquet(parquet_path)
    if "permno" in raw.columns:
        first_permno = raw["permno"].dropna().iloc[0]
        raw = raw.loc[raw["permno"] == first_permno]
    adapted = adapt_crsp_dsf_v2(raw, date_col="date" if "date" in raw.columns else None)
    value = evaluate_last_available("crsp", adapted, CRSP_EXPR, min_history=21)
    return {"crsp_value": value, "crsp_rows": float(adapted.shape[0])}


def fetch_wrds_crsp_dsf_v2_sample(
    *,
    permno: int = 14593,
    start_date: str = "2023-01-01",
    end_date: str = "2023-03-31",
) -> pd.DataFrame:
    """Fetch a tiny CRSP ``dsf_v2`` sample from WRDS for one PERMNO.

    The default PERMNO is Apple. This function intentionally does not print the
    WRDS username or any credential-derived values.
    """

    load_dotenv()
    userid = os.getenv("WRDS_USERID")
    if not userid:
        raise RuntimeError("WRDS_USERID is not set in the environment or .env")
    password = normalize_wrds_password(
        os.getenv("WRDS_PASSWORD") or os.getenv("PGPASSWORD") or os.getenv("WRDS_PGPASS")
    )
    if not password:
        raise RuntimeError(
            "WRDS_PASSWORD/PGPASSWORD/WRDS_PGPASS is not set; live WRDS smoke tests need "
            "explicit non-interactive credentials"
        )

    try:
        import wrds
    except ImportError as exc:
        raise RuntimeError("Python package 'wrds' is required for live WRDS queries") from exc

    columns_sql = ", ".join(WRDS_CRSP_COLUMNS)
    query = f"""
        select {columns_sql}
        from crsp.dsf_v2
        where permno = %(permno)s
          and dlycaldt between %(start_date)s and %(end_date)s
        order by dlycaldt
    """
    params = {"permno": int(permno), "start_date": start_date, "end_date": end_date}

    connection_kwargs = {"wrds_username": userid}
    if password:
        connection_kwargs["wrds_password"] = password

    try:
        db = wrds.Connection(**connection_kwargs)
    except EOFError as exc:
        raise RuntimeError(
            "WRDS attempted an interactive login. Add WRDS_PASSWORD/PGPASSWORD "
            "to the environment, or configure a working ~/.pgpass, then rerun."
        ) from exc
    try:
        return db.raw_sql(query, params=params, date_cols=["dlycaldt"])
    finally:
        db.close()


def run_wrds_crsp_smoke(
    *,
    permno: int = 14593,
    start_date: str = "2023-01-01",
    end_date: str = "2023-03-31",
) -> dict[str, float]:
    """Fetch a tiny WRDS CRSP sample and run one DSL expression on it."""

    raw = fetch_wrds_crsp_dsf_v2_sample(permno=permno, start_date=start_date, end_date=end_date)
    adapted = adapt_crsp_dsf_v2(raw, date_col="dlycaldt")
    value = evaluate_last_available("crsp", adapted, CRSP_EXPR, min_history=21)
    return {"wrds_crsp_value": value, "wrds_crsp_rows": float(adapted.shape[0])}


def wrds_env_status() -> dict[str, bool]:
    """Return whether required WRDS environment variables are present."""

    load_dotenv()
    return {"wrds_userid_present": bool(os.getenv("WRDS_USERID"))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Factor Lab DSL smoke checks.")
    parser.add_argument("--crypto-panel", type=Path, default=DEFAULT_CRYPTO_PANEL)
    parser.add_argument("--crypto-ticker", default="BTC-USD")
    parser.add_argument("--crsp-parquet", type=Path, default=DEFAULT_CRSP_PARQUET)
    parser.add_argument("--skip-crsp", action="store_true")
    parser.add_argument("--wrds-crsp", action="store_true", help="query a tiny live WRDS CRSP sample")
    parser.add_argument("--wrds-permno", type=int, default=14593)
    parser.add_argument("--wrds-start", default="2023-01-01")
    parser.add_argument("--wrds-end", default="2023-03-31")
    args = parser.parse_args()

    print(f"WRDS_USERID present: {wrds_env_status()['wrds_userid_present']}")

    crypto = run_crypto_smoke(args.crypto_panel, args.crypto_ticker)
    print(f"crypto smoke: rows={int(crypto['crypto_rows'])}, value={crypto['crypto_value']:.6f}")

    if not args.skip_crsp and args.crsp_parquet is not None and args.crsp_parquet.exists():
        try:
            crsp = run_crsp_local_smoke(args.crsp_parquet)
        except ImportError as exc:
            print(f"crsp smoke skipped: optional parquet engine missing ({exc.__class__.__name__})")
        except Exception as exc:
            print(f"crsp smoke skipped: {exc}")
        else:
            print(f"crsp smoke: rows={int(crsp['crsp_rows'])}, value={crsp['crsp_value']:.6f}")

    if args.wrds_crsp:
        try:
            wrds_crsp = run_wrds_crsp_smoke(
                permno=args.wrds_permno,
                start_date=args.wrds_start,
                end_date=args.wrds_end,
            )
        except RuntimeError as exc:
            print(f"wrds crsp smoke failed: {exc}")
            return 1
        print(
            "wrds crsp smoke: "
            f"rows={int(wrds_crsp['wrds_crsp_rows'])}, "
            f"value={wrds_crsp['wrds_crsp_value']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
