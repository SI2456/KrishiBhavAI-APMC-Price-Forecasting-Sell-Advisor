"""How often does each market report each commodity? Answer this before
downloading everything by hand.

Uses data.gov.in's "Variety-wise Daily Market Prices Data of Commodity", which
holds the Agmarknet price history (prices only, no arrivals). For every market
and commodity it downloads all rows, saves them as
data/raw/agmarknet/datagov_hist/<commodity>_<market>_datagov.csv (re-used on the
next run, and loaded as a price backfill by `load-files`), then reports the share
of trading days (Mon-Sat) with a price, per year.
"""
from __future__ import annotations

import logging
import re

import pandas as pd
import requests

from . import config, datagov
from .clean import norm_market, normalise

log = logging.getLogger(__name__)
API = "https://api.data.gov.in/resource/{rid}"
PAGE = 1000
FIELDS = ["State", "District", "Market", "Commodity", "Variety", "Grade", "Arrival_Date",
          "Min_Price", "Max_Price", "Modal_Price"]


def hist_dir():
    return config.RAW_AGMARKNET / "datagov_hist"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fetch_rows(api_key: str, market: str, commodity: str, session: requests.Session) -> list[dict]:
    """Every row for one exact market and commodity name (pages come in no date order)."""
    url = API.format(rid=config.DATAGOV_HIST_RESOURCE_ID)
    rows: list[dict] = []
    offset = 0
    while True:
        params = {"api-key": api_key, "format": "json", "limit": PAGE, "offset": offset,
                  "filters[State]": config.STATE, "filters[Market]": market,
                  "filters[Commodity]": commodity}
        payload = datagov._get(session, url, params)
        batch = payload.get("records", []) or []
        rows.extend(batch)
        offset += PAGE
        if not batch or offset >= int(payload.get("total", 0) or 0):
            return rows


def download(markets: dict[str, str], refresh: bool = False) -> pd.DataFrame:
    """markets: canonical name -> Agmarknet name. Returns all raw rows."""
    if not config.DATAGOV_API_KEY:
        raise RuntimeError("Set DATAGOV_API_KEY in .env (free key from data.gov.in → My Account).")
    session = requests.Session()
    session.headers["User-Agent"] = datagov.USER_AGENT
    hist_dir().mkdir(parents=True, exist_ok=True)
    frames = []
    for canon_market, raw_market in markets.items():
        for commodity, raw_names in config.AGMARKNET_COMMODITY_NAMES.items():
            path = hist_dir() / f"{_slug(commodity)}_{_slug(canon_market)}_datagov.csv"
            if path.exists() and not refresh:
                frames.append(pd.read_csv(path, dtype=str))
                continue
            rows = []
            for name in raw_names:
                for market_name in (raw_market, f"{raw_market} APMC"):
                    rows += fetch_rows(config.DATAGOV_API_KEY, market_name, name, session)
            frame = pd.DataFrame(rows, columns=FIELDS)
            frame.to_csv(path, index=False)
            log.info("%s at %s: %d rows -> %s", commodity, canon_market, len(frame), path.name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)


def coverage_by_year(clean: pd.DataFrame, start: str | None = None,
                     end: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per market, commodity and year: days with a price / trading days."""
    start_ts = pd.Timestamp(start or config.HISTORY_START)
    dates = pd.to_datetime(clean["date"])
    end_ts = end if end is not None else dates.max()
    rows = []
    for (market, commodity), s in clean.assign(date=dates).groupby(["market", "commodity"]):
        days = pd.DatetimeIndex(s["date"].unique())
        variety_days = s.groupby("variety")["date"].nunique().sort_values(ascending=False)
        for year in range(start_ts.year, end_ts.year + 1):
            lo, hi = max(pd.Timestamp(year, 1, 1), start_ts), min(pd.Timestamp(year, 12, 31), end_ts)
            span = pd.date_range(lo, hi, freq="D")
            trading = span[span.dayofweek != 6]
            reported = days[(days >= lo) & (days <= hi) & (days.dayofweek != 6)]
            rows.append({"market": market, "commodity": commodity, "year": year,
                         "days": len(reported), "trading_days": len(trading),
                         "pct": round(100 * len(reported) / max(len(trading), 1), 1),
                         "main_variety": variety_days.index[0] if len(variety_days) else "",
                         "main_variety_share": round(100 * variety_days.iloc[0] / max(len(days), 1), 1)
                         if len(variety_days) else 0.0})
    return pd.DataFrame(rows)


def summary(by_year: pd.DataFrame) -> pd.DataFrame:
    """Wide table: % of trading days per year, plus the whole-period %."""
    if by_year.empty:
        return by_year
    wide = by_year.pivot_table(index=["market", "commodity"], columns="year", values="pct")
    totals = by_year.groupby(["market", "commodity"]).agg(
        days=("days", "sum"), trading=("trading_days", "sum"),
        variety=("main_variety", "first"), variety_pct=("main_variety_share", "first"))
    wide.insert(0, "all_%", (100 * totals["days"] / totals["trading"]).round(1))
    wide["main_variety"] = totals["variety"] + " (" + totals["variety_pct"].astype(str) + "% of days)"
    return wide.sort_values("all_%", ascending=False)


def check_markets(extra_markets: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    markets = dict(config.AGMARKNET_MARKET_NAMES)
    for raw in extra_markets or []:                 # e.g. "Tharad", "Thara(Shihori)"
        markets[norm_market(raw)] = raw
    raw_rows = download(markets, refresh=refresh)
    if raw_rows.empty:
        log.warning("No rows returned. Check the market names in AGMARKNET_MARKET_NAMES.")
        return pd.DataFrame()
    clean, rep = normalise(raw_rows, source="datagov_hist", file_hint="datagov_hist", keep_all=True)
    log.info(rep.line())
    by_year = coverage_by_year(clean)
    table = summary(by_year)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(config.REPORTS / "market_check_by_year.csv", index=False)
    table.to_csv(config.REPORTS / "market_check.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print("\n% of trading days (Mon-Sat) with a price, per year:\n")
        print(table.to_string(float_format=lambda v: f"{v:.0f}"))
    print(f"\nSaved {config.REPORTS / 'market_check.csv'}. Rows were cached in {hist_dir()}; "
          "`load-files` loads them as a price backfill (no arrivals).")
    return table
