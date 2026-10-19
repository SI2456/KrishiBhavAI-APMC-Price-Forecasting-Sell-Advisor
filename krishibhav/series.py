"""Turn raw price rows into one clean daily series per (market, commodity).

Several varieties/grades are often reported on the same day. They are combined
into one daily price, weighted by arrivals when arrivals are known, otherwise
the median modal price. Short gaps (Sundays, holidays) are forward-filled up to
MAX_FILL_DAYS days and flagged, longer gaps stay empty.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, db

log = logging.getLogger(__name__)


def read_prices(engine=None) -> pd.DataFrame:
    engine = engine or db.get_engine()
    db.init_db(engine)
    with engine.connect() as conn:
        df = pd.read_sql_table("mandi_prices", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def combine_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per date, market, commodity."""
    if df.empty:
        return df
    d = df.copy()
    d["w"] = d["arrivals_tonnes"].where(d["arrivals_tonnes"] > 0, 0.0).fillna(0.0)
    d["mw"] = d["modal_price"] * d["w"]
    g = d.groupby(["market", "commodity", "date"])
    out = pd.DataFrame({
        "sum_mw": g["mw"].sum(),
        "sum_w": g["w"].sum(),
        "median_modal": g["modal_price"].median(),
        "min_price": g["min_price"].min(),
        "max_price": g["max_price"].max(),
        "arrivals_tonnes": g["arrivals_tonnes"].sum(min_count=1),
        "n_rows": g.size(),
    }).reset_index()
    out["modal_price"] = np.where(out["sum_w"] > 0, out["sum_mw"] / out["sum_w"].where(out["sum_w"] > 0),
                                  out["median_modal"])
    out["modal_price"] = out["modal_price"].round(2)
    return out.drop(columns=["sum_mw", "sum_w", "median_modal"])


def fill_calendar(daily: pd.DataFrame, max_fill: int = config.MAX_FILL_DAYS) -> pd.DataFrame:
    """Reindex each series to every calendar day and fill short gaps only."""
    parts = []
    for (market, commodity), s in daily.groupby(["market", "commodity"]):
        s = s.set_index("date").sort_index()
        idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(idx)
        s["reported"] = s["modal_price"].notna()
        for col in ["modal_price", "min_price", "max_price"]:
            s[col] = s[col].ffill(limit=max_fill)
        s["filled"] = ~s["reported"] & s["modal_price"].notna()
        s["market"], s["commodity"] = market, commodity
        parts.append(s.rename_axis("date").reset_index())
    if not parts:
        return daily
    cols = ["date", "market", "commodity", "modal_price", "min_price", "max_price",
            "arrivals_tonnes", "n_rows", "reported", "filled"]
    return pd.concat(parts, ignore_index=True)[cols]


def build_series() -> pd.DataFrame:
    prices = read_prices()
    if prices.empty:
        log.warning("No prices in the database yet. Run load-files or fetch-today first.")
        return prices
    series = fill_calendar(combine_daily(prices))
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "daily_series.csv"
    series.to_csv(out, index=False)
    log.info("Wrote %d rows (%d series) to %s", len(series),
             series.groupby(["market", "commodity"]).ngroups, out)
    return series
