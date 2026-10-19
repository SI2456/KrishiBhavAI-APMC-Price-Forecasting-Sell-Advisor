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
PROCESSED = DATA / "processed"
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

STATE = "Gujarat"
HISTORY_START = "2015-01-01"

# Canonical commodity name -> lowercase words that identify it in raw files.
# Each alias must appear as a whole word ("amaranth" does not match "Amaranthus").
# Order matters: the first match wins.
COMMODITIES: dict[str, list[str]] = {
    "Potato": ["potato", "aloo", "bataka"],
    "Castor Seed": ["castor", "erandi"],
    "Groundnut": ["groundnut", "ground nut", "peanut"],
    # Fourth crop: keep the one your father's shop handles most, delete the others.
    "Rajgira": ["rajgir", "rajgira", "amaranth"],
    "Bajra": ["bajra", "pearl millet"],
    "Cumin": ["cumin", "jeera"],
}

# Raw names that contain an alias above but are a different product with a
# different price, e.g. "Sweet Potato", "Groundnut Oil", "Groundnut Seed"
# (shelled kernels, priced well above pods). Matched as whole words.
NOT_COMMODITY: dict[str, list[str]] = {
    "Potato": ["sweet"],
    "Castor Seed": ["oil", "cake"],
    "Groundnut": ["oil", "cake", "seed", "split", "kernel", "kernels"],
}

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
