"""Database tables and an upsert helper (works on SQLite and PostgreSQL)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.engine import Engine

from . import config

metadata = MetaData()

# One row per (date, market, commodity, variety, grade) as Agmarknet reports it.
mandi_prices = Table(
    "mandi_prices",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", Date, nullable=False),
    Column("state", String(64)),
    Column("district", String(64)),
    Column("market", String(96), nullable=False),
    Column("commodity", String(64), nullable=False),
    Column("variety", String(96), nullable=False),
    Column("grade", String(32), nullable=False),
    Column("min_price", Float),        # Rs / quintal
    Column("max_price", Float),        # Rs / quintal
    Column("modal_price", Float),      # Rs / quintal
    Column("arrivals_tonnes", Float),
    Column("commodity_raw", String(128)),
    Column("source", String(32)),      # agmarknet_file | ceda_file | datagov
    Column("loaded_at", DateTime),
    UniqueConstraint("date", "market", "commodity", "variety", "grade", name="uq_price_row"),
)
PRICE_KEY = ["date", "market", "commodity", "variety", "grade"]

def _weather_cols() -> list[Column]:
    return [
        Column("precipitation_sum", Float),          # mm
        Column("temperature_2m_max", Float),         # deg C
        Column("temperature_2m_min", Float),
        Column("temperature_2m_mean", Float),
        Column("et0_fao_evapotranspiration", Float),  # mm
        Column("fetched_at", DateTime),
    ]


# Observed weather (ERA5 reanalysis). kind is always "history".
weather_daily = Table(
    "weather_daily",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("location", String(64), nullable=False),
    Column("date", Date, nullable=False),
    Column("kind", String(16), nullable=False),
    *_weather_cols(),
    UniqueConstraint("location", "date", "kind", name="uq_weather_row"),
)
WEATHER_KEY = ["location", "date", "kind"]

# Every 16-day forecast is kept with the day it was issued, so back-tests can use
# the forecast that was known at the time instead of the weather that happened.
weather_forecast = Table(
    "weather_forecast",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("location", String(64), nullable=False),
    Column("issued", Date, nullable=False),
    Column("date", Date, nullable=False),
    *_weather_cols(),
    UniqueConstraint("location", "issued", "date", name="uq_forecast_row"),
)
FORECAST_KEY = ["location", "issued", "date"]


def get_engine(url: str | None = None) -> Engine:
    url = url or config.DATABASE_URL
    if url.startswith("sqlite:///"):
        config.DATA.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _chunks(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def upsert(engine: Engine, table: Table, rows: list[dict], key_cols: list[str]) -> int:
    """Insert rows; on a key clash, update columns with the new non-null values.

    COALESCE(new, old) means loading a price file after an arrivals file (or the
    reverse) fills in both sets of columns instead of blanking one of them.
    """
    if not rows:
        return 0
    dialect = engine.dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise RuntimeError(f"Unsupported database: {dialect}")

    stmt = insert(table)
    update_cols = {
        c.name: func.coalesce(stmt.excluded[c.name], table.c[c.name])
        for c in table.columns
        if c.name not in key_cols and c.name != "id"
    }
    stmt = stmt.on_conflict_do_update(index_elements=key_cols, set_=update_cols)
    with engine.begin() as conn:
        for chunk in _chunks(rows, 500):
            conn.execute(stmt, chunk)
    return len(rows)
