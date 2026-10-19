"""Daily weather from Open-Meteo (free, no API key): history + 16-day forecast.

History goes to weather_daily. Each run re-fetches the last REFETCH_DAYS days,
because the newest days are early estimates that Open-Meteo later corrects.
Forecasts are saved first as data/raw/weather/forecast/<issue date>.json (the
GitHub job commits these every evening), then loaded into weather_forecast, one
set per issue date. Past forecasts cannot be downloaded later, so these files
are the only record of what was forecast on each day.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
from sqlalchemy import select, func

from . import config, db

log = logging.getLogger(__name__)
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 6   # the reanalysis archive runs a few days behind today
REFETCH_DAYS = 10      # re-download this many recent history days on every run


def _rows(payload: dict, location: str, **extra) -> list[dict]:
    """One dict per day; `extra` adds kind="history" or issued=<date>."""
    daily = payload.get("daily") or {}
    days = daily.get("time", [])
    fetched = db.now_utc()
    rows = []
    for i, d in enumerate(days):
        row = {"location": location, "date": date.fromisoformat(d), "fetched_at": fetched, **extra}
        for var in config.WEATHER_DAILY_VARS:
            values = daily.get(var) or []
            row[var] = values[i] if i < len(values) else None
        rows.append(row)
    return rows


def _call(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _base(lat: float, lon: float) -> dict:
    return {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "daily": ",".join(config.WEATHER_DAILY_VARS)}


def fetch_forecast_raw() -> Path:
    """Download today's 16-day forecast for every location and save the responses."""
    issued = date.today()
    payloads = {name: _call(FORECAST, {**_base(lat, lon), "forecast_days": 16})
                for name, (lat, lon) in config.WEATHER_LOCATIONS.items()}
    config.RAW_WEATHER_FORECAST.mkdir(parents=True, exist_ok=True)
    out = config.RAW_WEATHER_FORECAST / f"{issued.isoformat()}.json"
    out.write_text(json.dumps({"issued": issued.isoformat(), "locations": payloads}, indent=1),
                   encoding="utf-8")
    log.info("Saved %d-location forecast to %s", len(payloads), out)
    return out


def _forecast_rows(path: Path) -> list[dict]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    issued = date.fromisoformat(saved["issued"])
    rows = []
    for name, payload in saved["locations"].items():
        rows += _rows(payload, name, issued=issued)
    return rows


def load_forecasts_raw(folder: Path | None = None) -> int:
    """Re-load every saved forecast file (data/raw/weather/forecast/*.json)."""
    folder = folder or config.RAW_WEATHER_FORECAST
    engine = db.get_engine()
    db.init_db(engine)
    total = 0
    for path in sorted(folder.glob("*.json")):
        total += db.upsert(engine, db.weather_forecast, _forecast_rows(path), db.FORECAST_KEY)
    log.info("Loaded %d forecast rows from saved files", total)
    return total


def _last_history_date(engine, location: str) -> date | None:
    t = db.weather_daily
    with engine.connect() as conn:
        return conn.execute(
            select(func.max(t.c.date)).where(t.c.location == location, t.c.kind == "history")
        ).scalar()


def fetch_weather(history: bool = True, forecast: bool = True, load: bool = True) -> int:
    """History into the database, and today's forecast saved as JSON then loaded.
    With load=False only the forecast file is saved (used by the GitHub job)."""
    if not load:
        if forecast:
            fetch_forecast_raw()
        return 0
    engine = db.get_engine()
    db.init_db(engine)
    total = 0
    if history:
        end = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
        for name, (lat, lon) in config.WEATHER_LOCATIONS.items():
            last = _last_history_date(engine, name)
            start = (last - timedelta(days=REFETCH_DAYS - 1)) if last else date.fromisoformat(config.HISTORY_START)
            if start <= end:
                payload = _call(ARCHIVE, {**_base(lat, lon), "start_date": start.isoformat(),
                                          "end_date": end.isoformat()})
                rows = _rows(payload, name, kind="history")
                total += db.upsert(engine, db.weather_daily, rows, db.WEATHER_KEY)
                log.info("%s: %d history days (%s to %s)", name, len(rows), start, end)
    if forecast:
        rows = _forecast_rows(fetch_forecast_raw())
        total += db.upsert(engine, db.weather_forecast, rows, db.FORECAST_KEY)
        log.info("Loaded %d forecast rows", len(rows))
    return total
