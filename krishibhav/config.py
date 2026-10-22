"""Project settings: paths, commodities, markets and weather locations.

Edit COMMODITIES and MARKETS to change what the pipeline keeps.
Secrets (API key, database URL) go in a `.env` file at the project root,
never in this file. See `.env.example`.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_AGMARKNET = DATA / "raw" / "agmarknet"   # put downloaded Agmarknet / CEDA files here
RAW_DATAGOV = DATA / "raw" / "datagov"       # daily API responses are saved here as JSON
RAW_WEATHER_FORECAST = DATA / "raw" / "weather" / "forecast"   # one 16-day forecast file per day
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"               # hand-made MSP, calendar and crop-season CSVs
MODELS_DIR = ROOT / "models"                 # trained models from `advise` (not in git)
REPORTS = ROOT / "reports"


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA / 'krishibhav.db').as_posix()}")
DATAGOV_API_KEY = os.getenv("DATAGOV_API_KEY", "")
# "Current Daily Price of Various Commodities from Various Markets (Mandi)" on data.gov.in.
# If data.gov.in changes the resource, copy the new ID from the dataset page into .env.
DATAGOV_RESOURCE_ID = os.getenv("DATAGOV_RESOURCE_ID", "9ef84268-d588-465a-a308-a864a43d0070")
# "Variety-wise Daily Market Prices Data of Commodity": Agmarknet history since
# the 2000s (prices only, no arrivals). Used by `check-markets`.
DATAGOV_HIST_RESOURCE_ID = os.getenv("DATAGOV_HIST_RESOURCE_ID", "35985678-0d79-46b4-9ed6-6f13308a1d24")

STATE = "Gujarat"
# One year before the 2015 training start, so the 365-day lag exists from day one.
HISTORY_START = "2014-01-01"

# Canonical commodity name -> lowercase words that identify it in raw files.
# Each alias must appear as a whole word ("amaranth" does not match "Amaranthus").
# Order matters: the first match wins.
COMMODITIES: dict[str, list[str]] = {
    "Potato": ["potato", "aloo", "bataka"],
    "Castor Seed": ["castor", "erandi"],
    "Groundnut": ["groundnut", "ground nut", "peanut"],
    # Fourth crop. Chosen with check-markets (Oct 2026): bajra is reported on 78% of
    # trading days at Deesa since 2014; rajgira only from 2024, cumin is an Unjha crop.
    "Bajra": ["bajra", "pearl millet"],
}

# Raw names that contain an alias above but are a different product with a
# different price, e.g. "Sweet Potato", "Groundnut Oil", "Groundnut Seed"
# (shelled kernels, priced well above pods). Matched as whole words.
NOT_COMMODITY: dict[str, list[str]] = {
    "Potato": ["sweet"],
    "Castor Seed": ["oil", "cake"],
    "Groundnut": ["oil", "cake", "seed", "split", "kernel", "kernels"],
}

# Variety spellings to merge, per commodity: canonical name -> whole-word aliases.
# Fill this in after `report` shows the real names in your data, e.g.
#   "Potato": {"Desi": ["desi", "local"], "Jyoti": ["jyoti"]},
VARIETY_ALIASES: dict[str, dict[str, list[str]]] = {}

# The daily series uses ONE variety per market and commodity, so a change in the
# day's variety mix is not mistaken for a price move. By default the variety
# reported on the most days is used; set one here to choose it yourself.
MAIN_VARIETY: dict[str, str] = {}

# Canonical market name -> role. Raw names are normalised first
# (e.g. "Deesa(Deesa Veg Yard)" -> "Deesa Veg Yard"), then matched here.
MARKETS: dict[str, str] = {
    "Deesa": "primary",
    "Deesa Bhildi": "primary",
    "Deesa Veg Yard": "primary",
    "Palanpur": "comparison",
    "Dhanera": "comparison",
    "Unjha": "comparison",
}

# Yards modelled as one market: their rows stay separate in the database (one
# per yard) and are combined into one daily price when the series is built.
# Each Deesa yard alone reports on about half the trading days; together castor
# reaches ~70% and bajra ~78% (check-markets, Oct 2026).
MARKET_GROUPS: dict[str, list[str]] = {
    "Deesa": ["Deesa", "Deesa Bhildi", "Deesa Veg Yard"],
}

# Exact names Agmarknet uses, for querying its history. Recent rows add " APMC"
# to the market name; check-markets asks for both spellings.
AGMARKNET_MARKET_NAMES: dict[str, str] = {
    "Deesa": "Deesa",
    "Deesa Bhildi": "Deesa(Bhildi)",
    "Deesa Veg Yard": "Deesa(Deesa Veg Yard)",
    "Palanpur": "Palanpur",
    "Dhanera": "Dhanera",
    "Unjha": "Unjha",
}
AGMARKNET_COMMODITY_NAMES: dict[str, list[str]] = {
    "Potato": ["Potato"],
    "Castor Seed": ["Castor Seed"],
    "Groundnut": ["Groundnut"],
    "Bajra": ["Bajra(Pearl Millet/Cumbu)"],
}

# Approximate town-centre coordinates (lat, lon). Check them on a map and
# add the main potato / castor growing pockets of Banaskantha if you want.
WEATHER_LOCATIONS: dict[str, tuple[float, float]] = {
    "Deesa": (24.26, 72.19),
    "Palanpur": (24.17, 72.43),
    "Dhanera": (24.51, 72.02),
    "Unjha": (23.80, 72.39),
}

WEATHER_DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "et0_fao_evapotranspiration",
]

# Forward-fill a missing trading day only if the gap is this short.
MAX_FILL_DAYS = 3

# ---- Features and evaluation (M2 onward)
# Series the models forecast, and the nearby markets used as lead indicators.
FEATURE_MARKETS = ["Deesa"]
LEAD_MARKETS: dict[str, list[str]] = {"Deesa": ["Palanpur", "Dhanera"]}
HORIZONS = [7, 14, 30]          # days ahead
# Time-ordered splits (never random). 2014 is warm-up for the 365-day lag.
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2015-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}

# ---- Sell / hold advisor (M5). The app lets the user change these.
# Cost of keeping one quintal one more day, Rs. Potato: cold-storage rent
# (roughly Rs 250-300 per quintal for an ~8-month season) plus handling.
# Others: warehouse or home storage and small losses. CHECK WITH THE SHOP.
STORAGE_COST_RS_QTL_DAY: dict[str, float] = {
    "Potato": 1.2,
    "Castor Seed": 0.3,
    "Groundnut": 0.3,
    "Bajra": 0.3,
}
INTEREST_RATE = 0.12            # yearly rate on money the farmer owes while waiting
# Hold only if the expected net gain beats this share of today's price. The plan
# started at 3%; the 2023 back-test (valid year only) chose 1%.
HOLD_THRESHOLD = 0.01

# A modal price this many times above (or below) the median of the 15 reported
# days around it is treated as a bad entry (e.g. Rs/kg typed as Rs/quintal).
SPIKE_RATIO = 3.0
