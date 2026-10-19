"""Turn raw price rows into one clean daily series per (market, commodity).

Steps, in order:
0. merge_markets: yards listed in config.MARKET_GROUPS become one market
   (Deesa, Deesa Bhildi and Deesa Veg Yard -> Deesa); crops and markets not in
   config are left out.
1. pick_variety: keep ONE variety per series (spellings merged with
   config.VARIETY_ALIASES; chosen with config.MAIN_VARIETY or the variety
   reported on the most days). Mixing varieties would make a change in the
   day's variety mix look like a price move.
2. combine_daily: several grades on the same day become one price, weighted
   by arrivals when arrivals are known, otherwise the median modal price.
3. drop_spikes: modal prices far from their neighbours (unit typos) are blanked.
4. fill_calendar: short gaps (Sundays, holidays) are forward-filled up to
   MAX_FILL_DAYS days and flagged; longer gaps stay empty.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, db
from .clean import has_word

log = logging.getLogger(__name__)
PRICE_COLS = ["modal_price", "min_price", "max_price"]


def read_prices(engine=None) -> pd.DataFrame:
    engine = engine or db.get_engine()
    db.init_db(engine)
    with engine.connect() as conn:
        df = pd.read_sql_table("mandi_prices", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def merge_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Keep configured crops and markets, and rename grouped yards to their group."""
    if df.empty:
        return df
    d = df[df["commodity"].isin(config.COMMODITIES) & df["market"].isin(config.MARKETS)].copy()
    group_of = {yard: group for group, yards in config.MARKET_GROUPS.items() for yard in yards}
    d["yard"] = d["market"]
    d["market"] = d["market"].map(lambda m: group_of.get(m, m))
    return d


def canon_variety(commodity: str, raw: object) -> str:
    """Map a raw variety name to its canonical name from config.VARIETY_ALIASES."""
    name = "" if pd.isna(raw) else str(raw).strip()
    for canon, aliases in config.VARIETY_ALIASES.get(commodity, {}).items():
        if any(has_word(name.lower(), a) for a in aliases):
            return canon
    return name.title()


def pick_variety(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows of one variety per (market, commodity). For a merged market the
    choice is made per yard, so yards that name the variety differently all count."""
    if df.empty:
        return df
    d = df.copy()
    if "variety" not in d.columns:
        d["variety"] = ""
    d["variety"] = [canon_variety(c, v) for c, v in zip(d["commodity"], d["variety"])]
    place = "yard" if "yard" in d.columns else "market"
    parts = []
    for (market, commodity), s in d.groupby([place, "commodity"]):
        priced = s[s["modal_price"].notna()]
        main = config.MAIN_VARIETY.get(commodity)
        if main is not None and main not in set(s["variety"]):
            log.warning("%s at %s: MAIN_VARIETY %r not in the data; using the most reported one",
                        commodity, market, main)
            main = None
        if main is None:
            days = (priced if len(priced) else s).groupby("variety")["date"].nunique()
            main = days.sort_index().idxmax()
        keep = s[s["variety"] == main]
        if keep["variety"].nunique() < s["variety"].nunique():
            log.info("%s at %s: using variety %r (%d of %d rows)",
                     commodity, market, main, len(keep), len(s))
        parts.append(keep)
    return pd.concat(parts, ignore_index=True)


def combine_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per date, market, commodity."""
    if df.empty:
        return df
    d = df.copy()
    if "variety" not in d.columns:
        d["variety"] = ""
    # Weight only rows that have a price, so an arrivals-only row cannot pull
    # the weighted average towards zero.
    d["w"] = d["arrivals_tonnes"].where((d["arrivals_tonnes"] > 0) & d["modal_price"].notna(), 0.0)
    d["w"] = d["w"].fillna(0.0)
    d["mw"] = d["modal_price"].fillna(0.0) * d["w"]
    g = d.groupby(["market", "commodity", "date"])
    out = pd.DataFrame({
        "variety": g["variety"].first(),
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


def drop_spikes(daily: pd.DataFrame, ratio: float = config.SPIKE_RATIO, window: int = 15
                ) -> pd.DataFrame:
    """Blank prices more than `ratio` times above or below the median of the
    `window` reported days around them (usually a unit or typing error)."""
    if daily.empty:
        return daily
    parts = []
    removed = 0
    for (market, commodity), s in daily.groupby(["market", "commodity"]):
        s = s.sort_values("date").copy()
        median = s["modal_price"].rolling(window, center=True, min_periods=5).median()
        r = s["modal_price"] / median
        bad = (r > ratio) | (r < 1 / ratio)
        if bad.any():
            log.warning("%s at %s: blanked %d spike(s), e.g. %s", commodity, market, int(bad.sum()),
                        ", ".join(f"{d:%Y-%m-%d} Rs {p:,.0f}" for d, p in
                                  s.loc[bad, ["date", "modal_price"]].head(3).itertuples(index=False)))
        s.loc[bad, PRICE_COLS] = np.nan
        removed += int(bad.sum())
        parts.append(s)
    if removed:
        log.info("Blanked %d spike prices in total", removed)
    return pd.concat(parts, ignore_index=True)


def daily_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Raw database rows -> one checked price per day per market and commodity."""
    return drop_spikes(combine_daily(pick_variety(merge_markets(prices))))


def fill_calendar(daily: pd.DataFrame, max_fill: int = config.MAX_FILL_DAYS) -> pd.DataFrame:
    """Reindex each series to every calendar day and fill short gaps only."""
    parts = []
    for (market, commodity), s in daily.groupby(["market", "commodity"]):
        s = s.set_index("date").sort_index()
        variety = s["variety"].iloc[0] if "variety" in s.columns else ""
        idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(idx)
        s["reported"] = s["modal_price"].notna()
        for col in PRICE_COLS:
            s[col] = s[col].ffill(limit=max_fill)
        s["filled"] = ~s["reported"] & s["modal_price"].notna()
        s["market"], s["commodity"], s["variety"] = market, commodity, variety
        parts.append(s.rename_axis("date").reset_index())
    if not parts:
        return daily
    cols = ["date", "market", "commodity", "variety", "modal_price", "min_price", "max_price",
            "arrivals_tonnes", "n_rows", "reported", "filled"]
    return pd.concat(parts, ignore_index=True)[cols]


def build_series() -> pd.DataFrame:
    prices = read_prices()
    if prices.empty:
        log.warning("No prices in the database yet. Run load-files or fetch-today first.")
        return prices
    series = fill_calendar(daily_prices(prices))
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "daily_series.csv"
    series.to_csv(out, index=False)
    log.info("Wrote %d rows (%d series) to %s", len(series),
             series.groupby(["market", "commodity"]).ngroups, out)
    return series
