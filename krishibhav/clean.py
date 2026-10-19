"""Read messy Agmarknet / CEDA / data.gov.in tables and turn them into clean rows.

Handles the usual problems in these files:
- title rows above the real header, and .xls files that are really HTML
- different column names for the same thing ("Modal Price (Rs./Quintal)", "modal_price")
- commas in numbers, zero prices, dd/mm/yyyy dates
- names like "Deesa(Deesa Veg Yard)" and "Potato(Desi)"
- commodity or market missing from the table but present in the file name
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

REQUIRED = ["date", "market", "commodity"]
VALUE_COLS = ["modal_price", "arrivals_tonnes"]   # a file needs at least one of these
PRICE_COLS = ["min_price", "max_price", "modal_price"]
OUTPUT_COLS = [
    "date", "state", "district", "market", "commodity", "variety", "grade",
    "min_price", "max_price", "modal_price", "arrivals_tonnes", "commodity_raw", "source",
]


# ---------------------------------------------------------------- columns
def canon_col(name: object) -> str | None:
    """Map any raw column header to our column name, or None to drop it."""
    k = re.sub(r"[^a-z]", "", str(name).lower())
    if not k or k.startswith("unnamed") or k in {"slno", "sno", "srno"}:
        return None
    if k.endswith("id"):                 # state_id, market_id: numbers, not names
        return None
    if "unit" in k or "code" in k or "commoditytype" in k or "origin" in k:
        return None
    if "modal" in k:
        return "modal_price"
    if k.startswith("min") or "minimum" in k:
        return "min_price"
    if k.startswith("max") or "maximum" in k:
        return "max_price"
    if "date" in k:
        return "date"
    if "arrival" in k:
        return "arrivals_tonnes"
    if "commodity" in k:
        return "commodity"
    if "variety" in k:
        return "variety"
    if "grade" in k:
        return "grade"
    if "market" in k:
        return "market"
    if "district" in k:
        return "district"
    if "state" in k:
        return "state"
    return None


# ---------------------------------------------------------------- values
def norm_market(raw: object) -> str:
    """'Deesa(Deesa Veg Yard)' -> 'Deesa Veg Yard', 'DEESA (BHILDI)' -> 'Deesa Bhildi'."""
    s = str(raw).lower()
    s = re.sub(r"\bapmc\b", " ", s)
    s = re.sub(r"[()\[\],/_-]", " ", s)
    words = s.split()
    if len(words) > 1 and words[0] == words[1]:      # "deesa deesa veg yard"
        words = words[1:]
    name = " ".join(w.capitalize() for w in words)
    lookup = {m.lower(): m for m in config.MARKETS}
    return lookup.get(name.lower(), name)


def has_word(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def norm_commodity(raw: object) -> str | None:
    """'Potato(Desi)' -> 'Potato'. Aliases match whole words only, and names listed
    in config.NOT_COMMODITY ('Sweet Potato', 'Groundnut Oil') are rejected."""
    s = str(raw).lower()
    for canon, aliases in config.COMMODITIES.items():
        if any(has_word(s, a) for a in aliases):
            if any(has_word(s, x) for x in config.NOT_COMMODITY.get(canon, [])):
                return None
            return canon
    return None


def to_number(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(",", "", regex=False).str.strip()
    out = pd.to_numeric(s, errors="coerce").astype("float64")
    return out.mask(out <= 0)          # 0 or negative price = not reported


def to_date(series: pd.Series) -> pd.Series:
    """Year-first dates ('2020-02-01', Excel's '2020-02-01 00:00:00') are read as
    year-month-day; everything else ('01/02/2020', '1 Feb 2020') as day first.
    dayfirst=True on its own turns '2020-02-01' into 2 January."""
    s = series.astype("string").str.strip()
    ymd = s.str.extract(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
    year_first = pd.to_datetime(ymd[0] + "-" + ymd[1].str.zfill(2) + "-" + ymd[2].str.zfill(2),
                                errors="coerce", format="%Y-%m-%d")
    day_first = pd.to_datetime(s.where(ymd[0].isna()), errors="coerce", dayfirst=True, format="mixed")
    return year_first.fillna(day_first).dt.date


def clean_text(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.replace({"nan": "", "None": "", "NA": "", "-": ""})


# ---------------------------------------------------------------- reading
def _find_header(raw: pd.DataFrame) -> int | None:
    """Index of the row that holds column names: the first row where several
    cells are recognised headers (a title like 'Commodity wise price report'
    is one cell, so it does not count)."""
    for i in range(min(len(raw), 25)):
        targets = {canon_col(v) for v in raw.iloc[i].tolist()} - {None}
        if targets & set(VALUE_COLS) and len(targets) >= 2:
            return i
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read csv / xls / xlsx / html exports, including Agmarknet's HTML-as-.xls."""
    suffix = path.suffix.lower()
    frames: list[pd.DataFrame] = []
    if suffix == ".csv":
        frames = [pd.read_csv(path, header=None, dtype=str, encoding_errors="replace")]
    elif suffix in {".xls", ".xlsx"}:
        try:
            frames = [pd.read_excel(path, header=None, dtype=str)]
        except Exception:                 # Agmarknet often serves HTML named .xls
            frames = pd.read_html(path)
    elif suffix in {".html", ".htm"}:
        frames = pd.read_html(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    for raw in sorted(frames, key=len, reverse=True):
        raw = raw.astype("string")
        if isinstance(raw.columns, pd.MultiIndex) or _find_header(raw) is None:
            # read_html may already have used the header row as columns
            cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in raw.columns]
            if {canon_col(c) for c in cols} & set(VALUE_COLS):
                raw.columns = cols
                return raw.reset_index(drop=True)
            continue
        h = _find_header(raw)
        table = raw.iloc[h + 1 :].copy()
        table.columns = [str(c) for c in raw.iloc[h].tolist()]
        return table.reset_index(drop=True)
    raise ValueError(f"No price table found in {path.name} (no 'Modal Price' or 'Arrivals' header)")


# ---------------------------------------------------------------- cleaning
@dataclass
class CleanReport:
    file: str
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict = field(default_factory=dict)
    modal_outside_range: int = 0
    min_max_swapped: int = 0

    def line(self) -> str:
        drops = ", ".join(f"{k}: {v}" for k, v in self.dropped.items() if v) or "none"
        return (f"{self.file}: {self.rows_in} rows in, {self.rows_out} kept "
                f"(dropped {drops}; min/max swapped: {self.min_max_swapped}; "
                f"modal outside min-max: {self.modal_outside_range})")


def normalise(df: pd.DataFrame, source: str, file_hint: str = "", keep_all: bool = False
              ) -> tuple[pd.DataFrame, CleanReport]:
    """Rename columns, clean values and keep configured commodities/markets."""
    rep = CleanReport(file=file_hint or source, rows_in=len(df))

    renamed: dict[str, str] = {}
    for col in df.columns:
        target = canon_col(col)
        if target and target not in renamed.values():
            renamed[col] = target
    df = df[list(renamed)].rename(columns=renamed).copy()

    # Fill commodity / market from the file name if the table lacks them,
    # e.g. "potato_deesa-veg-yard_2015-2026.xlsx".
    hint = file_hint.lower().replace("_", " ").replace("-", " ")
    if "commodity" not in df.columns:
        guess = norm_commodity(hint)
        if guess:
            df["commodity"] = guess
    if "market" not in df.columns:
        for m in sorted(config.MARKETS, key=len, reverse=True):
            if m.lower() in hint:
                df["market"] = m
                break

    missing = [c for c in REQUIRED if c not in df.columns]
    if not any(c in df.columns for c in VALUE_COLS):
        missing.append("modal_price or arrivals_tonnes")
    if missing:
        raise ValueError(
            f"{rep.file}: missing columns {missing}. Columns found: {list(df.columns)}. "
            "Name the file with commodity and market (e.g. potato_deesa_2015-2026.xlsx) "
            "or export the report with those columns."
        )

    for col in ["state", "district", "variety", "grade", "arrivals_tonnes", *PRICE_COLS]:
        if col not in df.columns:
            df[col] = pd.NA

    df["commodity_raw"] = clean_text(df["commodity"])
    df["commodity"] = df["commodity_raw"].map(norm_commodity)
    if keep_all:
        df["commodity"] = df["commodity"].fillna(df["commodity_raw"])
    df["market"] = clean_text(df["market"]).map(norm_market)
    df["state"] = clean_text(df["state"]).replace("", config.STATE).str.title()
    df["district"] = clean_text(df["district"]).str.title()
    df["variety"] = clean_text(df["variety"]).str.title()
    df["grade"] = clean_text(df["grade"]).str.upper()
    for col in PRICE_COLS:
        df[col] = to_number(df[col])
    arrivals = pd.to_numeric(
        df["arrivals_tonnes"].astype("string").str.replace(",", "", regex=False), errors="coerce"
    ).astype("float64")
    df["arrivals_tonnes"] = arrivals.mask(arrivals < 0)
    # Some rows have min and max typed the wrong way round.
    flipped = df["min_price"] > df["max_price"]
    df.loc[flipped, ["min_price", "max_price"]] = df.loc[flipped, ["max_price", "min_price"]].to_numpy()
    rep.min_max_swapped = int(flipped.sum())
    df["date"] = to_date(df["date"])
    df["source"] = source

    before = len(df)
    df = df[df["date"].notna()]
    rep.dropped["no date"] = before - len(df)
    before = len(df)
    # Rows with arrivals but no price are kept: a separate arrivals report
    # merges into the price rows with the same key when loaded.
    df = df[df["modal_price"].notna() | df["arrivals_tonnes"].notna()]
    rep.dropped["no price or arrivals"] = before - len(df)
    before = len(df)
    df = df[df["commodity"].notna() & (df["commodity"] != "")]
    rep.dropped["other commodity"] = before - len(df)
    if not keep_all:
        before = len(df)
        df = df[df["market"].isin(config.MARKETS)]
        rep.dropped["other market"] = before - len(df)

    outside = (df["min_price"].notna() & df["max_price"].notna()
               & ((df["modal_price"] < df["min_price"]) | (df["modal_price"] > df["max_price"])))
    rep.modal_outside_range = int(outside.sum())

    before = len(df)
    df = df.drop_duplicates(subset=["date", "market", "commodity", "variety", "grade"], keep="last")
    rep.dropped["duplicates"] = before - len(df)

    df = df[OUTPUT_COLS].reset_index(drop=True)
    rep.rows_out = len(df)
    return df, rep


def to_records(df: pd.DataFrame, loaded_at) -> list[dict]:
    """DataFrame -> list of plain dicts ready for the database (NaN -> None)."""
    out = df.astype(object).where(df.notna(), None)
    records = out.to_dict(orient="records")
    for r in records:
        r["loaded_at"] = loaded_at
        for k, v in r.items():
            if isinstance(v, (np.floating,)):
                r[k] = float(v)
            if v is pd.NA:
                r[k] = None
    return records
