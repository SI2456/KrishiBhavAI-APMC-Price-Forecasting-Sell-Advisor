"""Daily weather from Open-Meteo (free, no API key): history + 16-day forecast."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import requests
from sqlalchemy import select, func

from . import config, db

log = logging.getLogger(__name__)
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 6   # the reanalysis archive runs a few days behind today


def _rows(payload: dict, location: str, kind: str) -> list[dict]:
    daily = payload.get("daily") or {}
    days = daily.get("time", [])
    fetched = db.now_utc()
    rows = []
    for i, d in enumerate(days):
        row = {"location": location, "date": date.fromisoformat(d), "kind": kind, "fetched_at": fetched}
        for var in config.WEATHER_DAILY_VARS:
            values = daily.get(var) or []
            row[var] = values[i] if i < len(values) else None
        rows.append(row)
    return rows


def _call(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _last_history_date(engine, location: str) -> date | None:
    t = db.weather_daily
    with engine.connect() as conn:
        return conn.execute(
            select(func.max(t.c.date)).where(t.c.location == location, t.c.kind == "history")
        ).scalar()


def fetch_weather(history: bool = True, forecast: bool = True) -> int:
    engine = db.get_engine()
    db.init_db(engine)
    total = 0
    end = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    for name, (lat, lon) in config.WEATHER_LOCATIONS.items():
        base = {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
                "daily": ",".join(config.WEATHER_DAILY_VARS)}
        if history:
            last = _last_history_date(engine, name)
            start = (last + timedelta(days=1)) if last else date.fromisoformat(config.HISTORY_START)
            if start <= end:
                payload = _call(ARCHIVE, {**base, "start_date": start.isoformat(), "end_date": end.isoformat()})
                rows = _rows(payload, name, "history")
                total += db.upsert(engine, db.weather_daily, rows, db.WEATHER_KEY)
                log.info("%s: %d history days (%s to %s)", name, len(rows), start, end)
        if forecast:
            payload = _call(FORECAST, {**base, "forecast_days": 16})
            rows = _rows(payload, name, "forecast")
            total += db.upsert(engine, db.weather_daily, rows, db.WEATHER_KEY)
            log.info("%s: %d forecast days", name, len(rows))
    return total
