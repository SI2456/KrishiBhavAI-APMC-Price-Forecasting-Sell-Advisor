"""Load downloaded Agmarknet / CEDA files and saved data.gov.in JSON into the database."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import config, db
from .clean import normalise, read_table, to_records

log = logging.getLogger(__name__)
FILE_TYPES = {".csv", ".xls", ".xlsx", ".html", ".htm"}


def load_files(folder: Path | None = None, keep_all: bool = False) -> int:
    """Clean every file in data/raw/agmarknet (and subfolders) and upsert it."""
    folder = folder or config.RAW_AGMARKNET
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in FILE_TYPES)
    if not files:
        log.warning("No files in %s. Download Agmarknet/CEDA reports there first.", folder)
        return 0

    engine = db.get_engine()
    db.init_db(engine)
    total = 0
    for path in files:
        name = path.name.lower()
        source = "ceda_file" if "ceda" in name else "datagov_hist" if "datagov" in name else "agmarknet_file"
        try:
            df, rep = normalise(read_table(path), source=source, file_hint=path.stem, keep_all=keep_all)
        except Exception as exc:  # keep going; report the bad file
            log.error("SKIPPED %s: %s", path.name, exc)
            continue
        total += db.upsert(engine, db.mandi_prices, to_records(df, db.now_utc()), db.PRICE_KEY)
        log.info(rep.line())
    log.info("Loaded %d rows from %d files into %s", total, len(files), engine.url)
    return total


def records_from_datagov_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload.get("records", []))


def load_datagov_raw(folder: Path | None = None) -> int:
    """Re-load every saved daily API response (data/raw/datagov/*.json)."""
    folder = folder or config.RAW_DATAGOV
    engine = db.get_engine()
    db.init_db(engine)
    total = 0
    for path in sorted(folder.glob("*.json")):
        frame = records_from_datagov_json(path)
        if frame.empty:
            continue
        df, rep = normalise(frame, source="datagov", file_hint=path.stem)
        total += db.upsert(engine, db.mandi_prices, to_records(df, db.now_utc()), db.PRICE_KEY)
        log.info(rep.line())
    log.info("Loaded %d rows from saved data.gov.in responses", total)
    return total
