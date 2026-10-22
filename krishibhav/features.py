"""M2: the feature table the models train on.

One row per day per (market, commodity) in config.FEATURE_MARKETS, with
features that use ONLY information available on that day (t) and targets for
t + h, h in config.HORIZONS:

  y_ret_<h>    log(P[t+h] / P[t])   the model target (scale-free across crops)
  y_price_<h>  P[t+h]               for reporting errors in Rs/quintal

Feature groups (column prefixes):
  price      log_price, ret_<k> (log return over k days), ma_gap_<w> (price vs its
             w-day mean), vol_<w> (std of daily log returns), spread (max-min)/modal,
             reported, days_since_report
  arrivals   arr, arr_7, arr_ratio_ly (7-day arrivals vs the same week last year);
             empty until an arrivals source is loaded
  weather    rain_<w>, tmean_30, tmax_7, et0_30 for the market's town
  policy     msp_rs_qtl, price_to_msp (empty for crops without MSP)
  calendar   dow, month, week, doy_sin/doy_cos, festival, apmc_holiday, policy,
             days_to_diwali, in_<phase> (sowing / harvest / ...)
  lead       x_<market>_gap (log price gap to a lead market), x_<market>_ret_7
  seasonal   seas_ret_<h>: last year's log change over the same h-day window
             (from the week around t-365 to the week around t+h-365); the
             seasonal-naive forecast, and a feature for the models

`split` marks train / valid / test from config.SPLITS (time order only),
`warmup` before training starts and `live` after the test period.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, db, reference, series

log = logging.getLogger(__name__)
LAGS = [1, 7, 14, 30, 365]
WINDOWS = [7, 30, 90]
ID_COLS = ["date", "market", "commodity", "split"]


# ---------------------------------------------------------------- inputs
def read_weather(engine=None) -> pd.DataFrame:
    engine = engine or db.get_engine()
    db.init_db(engine)
    with engine.connect() as conn:
        df = pd.read_sql_table("weather_daily", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["kind"] == "history"].drop(columns=["id", "kind", "fetched_at"], errors="ignore")


# ---------------------------------------------------------------- feature groups
def price_features(s: pd.DataFrame) -> pd.DataFrame:
    """s: one series on a continuous daily index (output of series.fill_calendar)."""
    p = s["modal_price"]
    lp = np.log(p)
    out = pd.DataFrame(index=s.index)
    out["log_price"] = lp
    for k in LAGS:
        if k == 365:
            # a year ago, averaged over a week so one missing day does not blank it
            out["ret_365"] = lp - lp.rolling(7, min_periods=1).mean().shift(362)
        else:
            out[f"ret_{k}"] = lp - lp.shift(k)
    r1 = lp.diff()
    for w in WINDOWS:
        out[f"ma_gap_{w}"] = p / p.rolling(w, min_periods=w // 2).mean() - 1
        out[f"vol_{w}"] = r1.rolling(w, min_periods=w // 2).std()
    out["spread"] = (s["max_price"] - s["min_price"]) / p
    out["spread_ma_7"] = out["spread"].rolling(7, min_periods=3).mean()
    reported = s["reported"].eq(True)
    out["reported"] = reported
    last = pd.Series(np.where(reported, np.arange(len(s)), np.nan), index=s.index).ffill()
    out["days_since_report"] = np.arange(len(s)) - last

    arr = s["arrivals_tonnes"]
    out["arr"] = arr
    out["arr_7"] = arr.rolling(7, min_periods=1).sum()
    out["arr_ratio_ly"] = out["arr_7"] / out["arr_7"].shift(364)

    # Week-averaged log price; the latest day it uses for seas_ret_h is
    # t + h - 365 + 3, which is in the past for every h up to 30.
    week = lp.rolling(7, center=True, min_periods=3).mean()
    for h in config.HORIZONS:
        out[f"seas_ret_{h}"] = week.shift(365 - h) - week.shift(365)
    return out


def targets(s: pd.DataFrame, horizons=None) -> pd.DataFrame:
    lp = np.log(s["modal_price"])
    out = pd.DataFrame(index=s.index)
    for h in horizons or config.HORIZONS:
        out[f"y_ret_{h}"] = lp.shift(-h) - lp
        out[f"y_price_{h}"] = s["modal_price"].shift(-h)
    return out


def weather_features(weather: pd.DataFrame, location: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    w = weather[weather["location"] == location].set_index("date").sort_index()
    out = pd.DataFrame(index=index)
    if w.empty:
        return out
    w = w.reindex(pd.date_range(min(w.index.min(), index.min()), max(w.index.max(), index.max()), freq="D"))
    rain = w["precipitation_sum"]
    for n in (7, 30, 90):
        out[f"rain_{n}"] = rain.rolling(n, min_periods=int(n * 0.8)).sum().reindex(index)
    out["tmean_30"] = w["temperature_2m_mean"].rolling(30, min_periods=24).mean().reindex(index)
    out["tmax_7"] = w["temperature_2m_max"].rolling(7, min_periods=6).max().reindex(index)
    out["et0_30"] = w["et0_fao_evapotranspiration"].rolling(30, min_periods=24).sum().reindex(index)
    return out


def calendar_features(index: pd.DatetimeIndex, crop: str, market: str, refs: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["dow"] = index.dayofweek
    out["month"] = index.month
    out["week"] = index.isocalendar().week.astype(int).to_numpy()
    doy = index.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    flags = reference.daily_flags(index, crop, market=market, msp=refs["msp"],
                                  calendar=refs["calendar"], seasons=refs["seasons"])
    return out.join(flags)


def lead_features(daily: pd.DataFrame, market: str, commodity: str, index: pd.DatetimeIndex,
                  own_log_price: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for lead in config.LEAD_MARKETS.get(market, []):
        s = daily[(daily["market"] == lead) & (daily["commodity"] == commodity)]
        name = lead.lower().replace(" ", "_")
        if s.empty:
            out[f"x_{name}_gap"] = np.nan
            out[f"x_{name}_ret_7"] = np.nan
            continue
        lp = np.log(s.set_index("date")["modal_price"].sort_index())
        lp = lp.reindex(pd.date_range(min(lp.index.min(), index.min()), max(lp.index.max(), index.max())))
        out[f"x_{name}_gap"] = own_log_price - lp.reindex(index)
        out[f"x_{name}_ret_7"] = (lp - lp.shift(7)).reindex(index)
    return out


def assign_split(dates: pd.Series) -> pd.Series:
    split = pd.Series("live", index=dates.index)
    first = min(pd.Timestamp(lo) for lo, _ in config.SPLITS.values())
    split[dates < first] = "warmup"
    for name, (lo, hi) in config.SPLITS.items():
        split[(dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))] = name
    return split


# ---------------------------------------------------------------- the table
def build_features(daily: pd.DataFrame, weather: pd.DataFrame, refs: dict | None = None,
                   markets: list[str] | None = None) -> pd.DataFrame:
    """daily: series.fill_calendar output for all markets (lead markets included)."""
    refs = refs or {"msp": reference.load_msp(), "calendar": reference.load_calendar(),
                    "seasons": reference.load_seasons()}
    markets = markets or config.FEATURE_MARKETS
    start = pd.Timestamp(config.HISTORY_START)
    parts = []
    for (market, commodity), s in daily[daily["market"].isin(markets)].groupby(["market", "commodity"]):
        s = s.set_index("date").sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))   # continuous days
        f = price_features(s)
        location = market if market in config.WEATHER_LOCATIONS else None
        if location:
            f = f.join(weather_features(weather, location, s.index))
        f = f.join(calendar_features(s.index, commodity, market, refs))
        f["price_to_msp"] = s["modal_price"] / f["msp_rs_qtl"]
        f = f.join(lead_features(daily, market, commodity, s.index, f["log_price"]))
        f = f.join(targets(s))
        f.insert(0, "commodity", commodity)
        f.insert(0, "market", market)
        f = f.rename_axis("date").reset_index()
        parts.append(f[(f["date"] >= start) & f["log_price"].notna()])   # rows with a price at t
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    for col in [c for c in table.columns if c.startswith("in_")]:
        table[col] = table[col].fillna(False).astype(bool)      # phases differ by crop
    table.insert(3, "split", assign_split(table["date"]))
    return table


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Rows per crop and split, and how often each target is available."""
    g = table.groupby(["commodity", "split"])
    out = g.size().rename("rows").to_frame()
    for h in config.HORIZONS:
        out[f"y_{h}_known_%"] = (100 * g[f"y_ret_{h}"].apply(lambda x: x.notna().mean())).round(1)
    return out


def build() -> pd.DataFrame:
    prices = series.read_prices()
    if prices.empty:
        log.warning("No prices in the database yet. Run load-files first.")
        return pd.DataFrame()
    daily = series.fill_calendar(series.daily_prices(prices))
    table = build_features(daily, read_weather())
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "features.csv"
    table.to_csv(out, index=False)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    nulls = (100 * table.notna().mean()).round(1).rename("non_null_%")
    nulls.to_csv(config.REPORTS / "feature_coverage.csv")
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(summary(table).to_string())
        empty = sorted(nulls[nulls == 0].index)
        if empty:
            print(f"\nEmpty columns (no source yet): {', '.join(empty)}")
    log.info("Wrote %d rows x %d columns to %s", len(table), table.shape[1], out)
    return table
