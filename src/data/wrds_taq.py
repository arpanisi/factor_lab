"""Build WRDS TAQ 1-minute bars/features for Factor Lab."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wrds_crsp import connect_wrds

TAQ_1M_COLUMNS = ["open", "high", "low", "close", "volume", "spread", "midret", "imbalance", "trade_size", "trade_count"]


def safe_symbol(symbol: str) -> str:
    """Return filename-safe TAQ symbol."""

    return str(symbol).replace("/", "_").replace(":", "_").replace(".", "_")


def taq_wct_table(date: str | pd.Timestamp, *, schema: str = "taqmsec") -> str:
    """Return the WRDS TAQ weighted/consolidated trade table for a date."""

    ts = pd.Timestamp(date)
    return f"{schema}.wct_{ts:%Y%m%d}"


def build_taq_wct_query(
    *,
    date: str | pd.Timestamp,
    symbols: Iterable[str],
    schema: str = "taqmsec",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
) -> tuple[str, dict[str, Any]]:
    """Build a parameterized query for one day of TAQ WCT rows."""

    symbol_tuple = tuple(str(symbol).upper().strip() for symbol in symbols if str(symbol).strip())
    if not symbol_tuple:
        raise ValueError("at least one TAQ symbol is required")
    table = taq_wct_table(date, schema=schema)
    query = f"""
        select date, time_m, sym_root, sym_suffix, size, price, nbo, nbb
        from {table}
        where sym_root = any(%(symbols)s)
          and time_m between %(start_time)s and %(end_time)s
          and price is not null
          and size is not null
        order by sym_root, time_m
    """
    return query, {"symbols": list(symbol_tuple), "start_time": start_time, "end_time": end_time}


def build_taq_symbol_check_query(
    *,
    date: str | pd.Timestamp,
    symbols: Iterable[str],
    schema: str = "taqmsec",
) -> tuple[str, dict[str, Any]]:
    """Build a query counting rows for candidate TAQ symbols on one day."""

    symbol_tuple = tuple(str(symbol).upper().strip() for symbol in symbols if str(symbol).strip())
    if not symbol_tuple:
        raise ValueError("at least one TAQ symbol is required")
    table = taq_wct_table(date, schema=schema)
    query = f"""
        select sym_root, count(*) as row_count
        from {table}
        where sym_root = any(%(symbols)s)
        group by sym_root
        order by sym_root
    """
    return query, {"symbols": list(symbol_tuple)}


def check_taq_symbols(
    *,
    symbols: tuple[str, ...],
    date: str,
    schema: str = "taqmsec",
) -> pd.DataFrame:
    """Return per-symbol TAQ row counts for one date."""

    query, params = build_taq_symbol_check_query(date=date, symbols=symbols, schema=schema)
    db = connect_wrds()
    try:
        out = db.raw_sql(query, params=params)
    finally:
        db.close()
    seen = set(out["sym_root"].astype(str).str.upper()) if not out.empty else set()
    missing = [{"sym_root": symbol.upper(), "row_count": 0} for symbol in symbols if symbol.upper() not in seen]
    if missing:
        out = pd.concat([out, pd.DataFrame(missing)], ignore_index=True)
    return out.sort_values("sym_root").reset_index(drop=True)


def fetch_taq_wct_day(
    db,
    *,
    date: str | pd.Timestamp,
    symbols: Iterable[str],
    schema: str = "taqmsec",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
) -> pd.DataFrame:
    """Fetch one day of TAQ WCT trade/NBBO rows."""

    query, params = build_taq_wct_query(
        date=date,
        symbols=symbols,
        schema=schema,
        start_time=start_time,
        end_time=end_time,
    )
    return db.raw_sql(query, params=params, date_cols=["date"])


def build_taq_1m_features(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate raw TAQ WCT rows into one 1-minute feature frame per symbol."""

    if raw.empty:
        return {}
    required = {"date", "time_m", "sym_root", "size", "price"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"TAQ rows missing columns: {missing}")

    df = raw.copy()
    df["sym_root"] = df["sym_root"].astype(str).str.upper().str.strip()
    df["timestamp"] = _parse_taq_timestamp(df["date"], df["time_m"])
    for col in ["price", "size", "nbo", "nbb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "sym_root", "price", "size"])
    df["mid"] = (df.get("nbo") + df.get("nbb")) / 2.0 if {"nbo", "nbb"} <= set(df.columns) else pd.NA
    df["spread_raw"] = (df.get("nbo") - df.get("nbb")) if {"nbo", "nbb"} <= set(df.columns) else pd.NA
    df["signed_size"] = _signed_trade_size(df)

    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in df.groupby("sym_root", sort=True):
        g = group.set_index("timestamp").sort_index()
        price = g["price"].resample("1min")
        out = pd.DataFrame(
            {
                "open": price.first(),
                "high": price.max(),
                "low": price.min(),
                "close": price.last(),
                "volume": g["size"].resample("1min").sum(),
                "spread": g["spread_raw"].resample("1min").mean(),
                "trade_size": g["size"].resample("1min").mean(),
                "trade_count": g["size"].resample("1min").count(),
                "signed_volume": g["signed_size"].resample("1min").sum(),
            }
        )
        out["imbalance"] = out["signed_volume"] / out["volume"].abs().where(out["volume"].abs() > 0)
        mid = g["mid"].resample("1min").last()
        out["midret"] = mid.pct_change()
        out = out.drop(columns=["signed_volume"])
        out = out[TAQ_1M_COLUMNS].replace([float("inf"), float("-inf")], pd.NA)
        out = out.dropna(subset=["open", "high", "low", "close", "volume"], how="any")
        frames[str(symbol)] = out
    return frames


def _signed_trade_size(df: pd.DataFrame) -> pd.Series:
    if "mid" not in df or df["mid"].isna().all():
        return pd.Series(0.0, index=df.index)
    sign = (df["price"] - df["mid"]).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    return sign * df["size"]


def _parse_taq_timestamp(date: pd.Series, time_m: pd.Series) -> pd.Series:
    """Parse TAQ date/time columns with mixed fractional-second formats."""

    values = date.astype(str) + " " + time_m.astype(str)
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except ValueError:
        return pd.to_datetime(values, errors="coerce")


def fetch_and_save_taq_1m(
    *,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    output_dir: Path,
    schema: str = "taqmsec",
    incremental: bool = True,
) -> dict[str, Any]:
    """Fetch TAQ WCT rows over a date range and save per-symbol 1-minute parquet files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start_date, end_date)
    collected: dict[str, list[pd.DataFrame]] = {symbol.upper(): [] for symbol in symbols}
    db = connect_wrds()
    try:
        for i, date in enumerate(dates, start=1):
            print(f"TAQ fetching {date:%Y-%m-%d} ({i}/{len(dates)})", flush=True)
            try:
                raw = fetch_taq_wct_day(db, date=date, symbols=symbols, schema=schema)
            except Exception as exc:
                print(f"TAQ day skipped {date:%Y-%m-%d}: {exc.__class__.__name__}: {exc}")
                continue
            print(f"TAQ raw rows {date:%Y-%m-%d}: {len(raw):,}", flush=True)
            day_frames = build_taq_1m_features(raw)
            for symbol, frame in day_frames.items():
                if incremental:
                    path = output_dir / f"{safe_symbol(symbol)}_1m.parquet"
                    _append_parquet_frame(path, frame)
                    print(f"TAQ wrote {symbol}: +{len(frame):,} rows -> {path}", flush=True)
                else:
                    collected.setdefault(symbol, []).append(frame)
    finally:
        db.close()

    summary = {"symbols": {}, "output_dir": str(output_dir)}
    for symbol in {item.upper() for item in symbols} | set(collected):
        path = output_dir / f"{safe_symbol(symbol)}_1m.parquet"
        if incremental:
            if not path.exists():
                summary["symbols"][symbol] = {"rows": 0, "path": ""}
                continue
            out = pd.read_parquet(path).sort_index()
        else:
            frames = collected.get(symbol, [])
            if not frames:
                summary["symbols"][symbol] = {"rows": 0, "path": ""}
                continue
            out = pd.concat(frames).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            out.to_parquet(path)
        summary["symbols"][symbol] = {
            "rows": int(out.shape[0]),
            "start": str(out.index.min()),
            "end": str(out.index.max()),
            "path": str(path),
        }
    return summary


def _append_parquet_frame(path: Path, frame: pd.DataFrame) -> None:
    """Append a frame to a parquet file by read/merge/write.

    TAQ benchmark slices are intentionally small enough for this simple and
    robust approach. It gives visible progress and preserves partial work.
    """

    if path.exists():
        existing = pd.read_parquet(path)
        out = pd.concat([existing, frame]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
    else:
        out = frame.sort_index()
    out.to_parquet(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch WRDS TAQ WCT data and save 1-minute bars.")
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA", help="comma-separated TAQ root symbols")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--schema", default="taqmsec")
    parser.add_argument("--output-dir", type=Path, default=Path("data/taq_1m"))
    parser.add_argument("--batch-save", action="store_true", help="save only at the end instead of after each day")
    parser.add_argument("--check-symbols-only", action="store_true", help="only count requested symbols on start-date")
    args = parser.parse_args(argv)

    symbols = tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
    if args.check_symbols_only:
        counts = check_taq_symbols(symbols=symbols, date=args.start_date, schema=args.schema)
        print(counts.to_string(index=False))
        return 0
    summary = fetch_and_save_taq_1m(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        schema=args.schema,
        incremental=not args.batch_save,
    )
    print("saved TAQ 1-minute files:")
    for symbol, info in summary["symbols"].items():
        print(f"{symbol}: rows={info['rows']} path={info['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
