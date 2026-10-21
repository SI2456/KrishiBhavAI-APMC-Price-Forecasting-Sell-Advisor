"""Hand-made reference tables in data/reference/ and the daily flags built from them.

  msp.csv           MSP per crop and kharif marketing season (Oct-Sep)
  calendar.csv      dated events: festivals, APMC holidays, policy shocks
  crop_seasons.csv  sowing / harvest windows that repeat every year

Each loader checks the file and raises one error that lists every problem, so
a typo in a hand-edited CSV is caught before it silently breaks the features.
daily_flags() turns the tables into one row of features per day for a crop.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, db

log = logging.getLogger(__name__)
EVENT_KINDS = {"festival", "apmc_holiday", "policy"}
STATUSES = {"verified", "check", "confirmed"}


def _read(name: str, path: Path | None, columns: list[str]) -> pd.DataFrame:
    path = path or config.REFERENCE / name
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False, skipinitialspace=True)
    df = df.apply(lambda c: c.str.strip())
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(df.columns)}")
    return df


def _fail(name: str, problems: list[str]) -> None:
    if problems:
        raise ValueError(f"{name} has {len(problems)} problem(s):\n  " + "\n  ".join(problems))


def _warn_unknown(name: str, df: pd.DataFrame, col: str, known) -> None:
    unknown = sorted(set(df[col]) - set(known) - {""})
    if unknown:
        log.warning("%s: %s not in config.py: %s (rows kept, but no series will use them)",
                    name, col, unknown)


# ---------------------------------------------------------------- MSP
def load_msp(path: Path | None = None) -> pd.DataFrame:
    df = _read("msp.csv", path, ["crop", "season", "msp_rs_qtl", "status"])
    problems = []
    first_year = pd.Series(np.nan, index=df.index)
    for i, r in df.iterrows():
        line = f"row {i + 1} ({r['crop']} {r['season']})"
        m = re.fullmatch(r"(\d{4})-(\d{2})", r["season"])
        if not m or (int(m.group(1)) + 1) % 100 != int(m.group(2)):
            problems.append(f"{line}: season must look like 2025-26")
        else:
            first_year[i] = int(m.group(1))
        value = pd.to_numeric(r["msp_rs_qtl"], errors="coerce")
        if not value > 0:
            problems.append(f"{line}: msp_rs_qtl must be a positive number, got {r['msp_rs_qtl']!r}")
        if r["status"] not in STATUSES:
            problems.append(f"{line}: status must be one of {sorted(STATUSES)}")
    dup = df[df.duplicated(["crop", "season"], keep=False)]
    for (crop, season), _ in dup.groupby(["crop", "season"]):
        problems.append(f"{crop} {season} is listed more than once")
    _fail("msp.csv", problems)
    _warn_unknown("msp.csv", df, "crop", config.COMMODITIES)

    df["msp_rs_qtl"] = df["msp_rs_qtl"].astype(float)
    df["valid_from"] = pd.to_datetime(first_year.astype(int).astype(str) + "-10-01")
    df["valid_to"] = df["valid_from"] + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    return df.sort_values(["crop", "valid_from"]).reset_index(drop=True)


def msp_on(dates, crop: str, msp: pd.DataFrame | None = None) -> pd.Series:
    """MSP in force on each date for a crop (NaN for crops without MSP or uncovered dates)."""
    msp = load_msp() if msp is None else msp
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    out = pd.Series(np.nan, index=idx, name="msp_rs_qtl")
    for r in msp[msp["crop"] == crop].itertuples():
        out[(idx >= r.valid_from) & (idx <= r.valid_to)] = r.msp_rs_qtl
    return out


# ---------------------------------------------------------------- dated events
def load_calendar(path: Path | None = None) -> pd.DataFrame:
    df = _read("calendar.csv", path, ["start", "end", "kind", "name", "crop", "market", "status"])
    problems = []
    start = pd.to_datetime(df["start"], format="%Y-%m-%d", errors="coerce")
    end = pd.to_datetime(df["end"].where(df["end"] != ""), format="%Y-%m-%d", errors="coerce").fillna(start)
    for i, r in df.iterrows():
        line = f"row {i + 1} ({r['name'] or r['kind']})"
        if pd.isna(start[i]):
            problems.append(f"{line}: start must be YYYY-MM-DD, got {r['start']!r}")
        elif r["end"] and pd.isna(pd.to_datetime(r["end"], format="%Y-%m-%d", errors="coerce")):
            problems.append(f"{line}: end must be YYYY-MM-DD, got {r['end']!r}")
        elif end[i] < start[i]:
            problems.append(f"{line}: end {r['end']} is before start {r['start']}")
        if r["kind"] not in EVENT_KINDS:
            problems.append(f"{line}: kind must be one of {sorted(EVENT_KINDS)}, got {r['kind']!r}")
        if not r["name"]:
            problems.append(f"{line}: name is empty")
        if r["status"] not in STATUSES:
            problems.append(f"{line}: status must be one of {sorted(STATUSES)}")
    _fail("calendar.csv", problems)
    _warn_unknown("calendar.csv", df, "crop", config.COMMODITIES)
    _warn_unknown("calendar.csv", df, "market", set(config.MARKETS) | set(config.MARKET_GROUPS))
    df["start"], df["end"] = start, end
    return df.sort_values("start").reset_index(drop=True)


# ---------------------------------------------------------------- recurring windows
def _month_day(value: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"(\d{2})-(\d{2})", value)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    try:
        pd.Timestamp(2024, month, day)                  # leap year, so 02-29 is allowed
    except ValueError:
        return None
    return month, day


def load_seasons(path: Path | None = None) -> pd.DataFrame:
    df = _read("crop_seasons.csv", path, ["crop", "phase", "start", "end", "status"])
    problems = []
    for i, r in df.iterrows():
        line = f"row {i + 1} ({r['crop']} {r['phase']})"
        for col in ("start", "end"):
            if _month_day(r[col]) is None:
                problems.append(f"{line}: {col} must be MM-DD, got {r[col]!r}")
        if not re.fullmatch(r"[a-z_]+", r["phase"]):
            problems.append(f"{line}: phase must be lowercase words joined by _, got {r['phase']!r}")
        if r["status"] not in STATUSES:
            problems.append(f"{line}: status must be one of {sorted(STATUSES)}")
    _fail("crop_seasons.csv", problems)
    _warn_unknown("crop_seasons.csv", df, "crop", config.COMMODITIES)
    return df


def _in_window(idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """True where the date's month-day falls in start..end (may wrap past 31 Dec)."""
    md = idx.month * 100 + idx.day
    (sm, sd), (em, ed) = _month_day(start), _month_day(end)
    lo, hi = sm * 100 + sd, em * 100 + ed
    return (md >= lo) & (md <= hi) if lo <= hi else (md >= lo) | (md <= hi)


# ---------------------------------------------------------------- features
def daily_flags(dates, crop: str, market: str | None = None, msp: pd.DataFrame | None = None,
                calendar: pd.DataFrame | None = None, seasons: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """One row per date: festival / APMC holiday / policy flags, days to the next
    Diwali, one in_<phase> column per crop phase, and the MSP in force."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    calendar = load_calendar() if calendar is None else calendar
    seasons = load_seasons() if seasons is None else seasons
    out = pd.DataFrame(index=idx)
    out.index.name = "date"

    applies = (calendar["crop"].isin(["", crop])
               & (calendar["market"].isin(["", market]) if market else calendar["market"].eq("")))
    events = calendar[applies]
    for kind in sorted(EVENT_KINDS):
        flag = np.zeros(len(idx), dtype=bool)
        for e in events[events["kind"] == kind].itertuples():
            flag |= (idx >= e.start) & (idx <= e.end)
        out[kind] = flag

    diwali = pd.DatetimeIndex(sorted(calendar.loc[calendar["name"].str.lower() == "diwali", "start"]))
    pos = diwali.searchsorted(idx)                          # first Diwali on or after each date
    out["days_to_diwali"] = [float((diwali[p] - d).days) if p < len(diwali) else np.nan
                             for d, p in zip(idx, pos)]

    for phase, rows in seasons[seasons["crop"] == crop].groupby("phase"):
        flag = np.zeros(len(idx), dtype=bool)
        for r in rows.itertuples():
            flag |= _in_window(idx, r.start, r.end)
        out[f"in_{phase}"] = flag

    out["msp_rs_qtl"] = msp_on(idx, crop, msp).to_numpy()
    return out


# ---------------------------------------------------------------- check command
def closed_day_candidates(prices: pd.DataFrame) -> pd.DataFrame:
    """Mon-Sat dates on which none of the configured markets reported any crop:
    likely APMC holidays (or days missing from Agmarknet)."""
    p = prices[prices["market"].isin(config.MARKETS) & prices["modal_price"].notna()]
    if p.empty:
        return pd.DataFrame(columns=["date", "weekday"])
    reported = pd.DatetimeIndex(pd.to_datetime(p["date"]).unique())
    lo = max(reported.min(), pd.Timestamp(config.HISTORY_START))
    days = pd.date_range(lo, reported.max(), freq="D")
    closed = days[(days.dayofweek != 6) & ~days.isin(reported)]
    return pd.DataFrame({"date": closed.date, "weekday": closed.day_name()})


def check_reference() -> None:
    msp, calendar, seasons = load_msp(), load_calendar(), load_seasons()
    print(f"msp.csv: {len(msp)} rows, crops {sorted(msp['crop'].unique())}, "
          f"seasons {msp['season'].min()} to {msp['season'].max()}, "
          f"{(msp['status'] == 'check').sum()} still marked 'check'")
    for crop in config.COMMODITIES:
        if crop not in set(msp["crop"]):
            print(f"  {crop}: no MSP rows (fine for crops without MSP, e.g. castor and potato)")
    print(f"calendar.csv: {len(calendar)} events, {calendar['start'].min():%Y-%m-%d} to "
          f"{calendar['end'].max():%Y-%m-%d}; by kind: {calendar['kind'].value_counts().to_dict()}")
    print(f"crop_seasons.csv: {len(seasons)} windows for {sorted(seasons['crop'].unique())}")

    from .series import read_prices
    prices = read_prices(db.get_engine())
    closed = closed_day_candidates(prices)
    if closed.empty:
        print("No prices in the database yet, so no holiday suggestions.")
        return
    known = calendar[calendar["kind"] == "apmc_holiday"]
    covered = np.zeros(len(closed), dtype=bool)
    closed_ts = pd.to_datetime(closed["date"])
    for e in known.itertuples():
        covered |= ((closed_ts >= e.start) & (closed_ts <= e.end)).to_numpy()
    closed["in_calendar"] = covered
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS / "closed_day_candidates.csv"
    closed.to_csv(out, index=False)
    by_year = closed.assign(year=closed_ts.dt.year).groupby("year")["in_calendar"].agg(["size", "sum"])
    print(f"\nMon-Sat days when none of {sorted(config.MARKETS)} reported anything "
          "(likely holidays), and how many the calendar already explains:")
    print(by_year.rename(columns={"size": "closed_days", "sum": "in_calendar"}).to_string())
    print(f"Saved the list to {out}; add the real holidays to calendar.csv.")
