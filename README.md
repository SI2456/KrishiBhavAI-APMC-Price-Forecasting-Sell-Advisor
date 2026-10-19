# KrishiBhav: Deesa APMC price forecasting

Week 1 code: collect, clean and store mandi prices and weather for Deesa APMC
and nearby markets, so the forecasting models (weeks 7 onward) have a clean
daily dataset to train on.

```
Agmarknet / CEDA files ─┐
data.gov.in daily API ──┼─► clean ─► database (SQLite or PostgreSQL) ─► daily_series.csv ─► models
Open-Meteo weather ─────┘                                            └► coverage report + charts
```

## 1. Setup (Windows)

```bat
git clone https://github.com/SI2456/KrishiBhavAI-APMC-Price-Forecasting-Sell-Advisor.git krishibhavAI
cd krishibhavAI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
pytest -q
```

Use Python 3.14 (the versions in `requirements.txt` are pinned to it).
On Mac/Linux use `source .venv/bin/activate` and `cp`. All 50 tests should pass.
They use fake files, so no internet is needed.

## 2. Download historical prices (one evening of clicking)

1. Open **agmarknet.gov.in → Price and Arrivals** (commodity-wise or market-wise report).
2. Choose State **Gujarat**, District **Banaskantha**, Market **Deesa** (also try
   **Deesa Bhildi** and **Deesa Veg Yard** for potato), the commodity, and a date range.
   If the site refuses long ranges, download one year at a time.
3. Export to Excel and save into `data/raw/agmarknet/` with a name like
   `potato_deesa-veg-yard_2015.xlsx` (see the README in that folder).
4. Repeat for each commodity (potato, castor, groundnut, your 4th crop) and
   the comparison markets (Palanpur, Dhanera, Unjha). Download the
   **arrivals** report too if it comes separately; rows with the same date,
   market, commodity, variety and grade merge automatically.

Faster alternative for the backfill: **CEDA Agri Market Data**
(agmarknet.ceda.ashoka.edu.in) lets you download Agmarknet data for a state in
one go. Put `ceda` in the file name.

Then load and check:

```bat
python -m krishibhav load-files
python -m krishibhav report
```

`report` prints coverage per market and commodity and saves a chart for each
in `reports/plots/`. **Look at every chart**: flat lines, sudden 10x jumps, or
long gaps mean a bad download. Coverage is counted from 2014 to the latest
date, so a series that only starts in 2022 shows low coverage. Under ~70% →
drop that series. The `variety` column shows which variety each series uses.

Start with one series (potato at Deesa Veg Yard), get it clean, then add the rest.

## 3. Daily live prices (data.gov.in)

1. Register at **data.gov.in** → My Account → copy your **API key**.
2. Put it in `.env`: `DATAGOV_API_KEY=your_key`
3. Run after 8 pm, when most yards have reported:

```bat
python -m krishibhav fetch-today
```

Each day's raw response is saved in `data/raw/datagov/` (the API only gives the
latest day, so these files become your live history).

**Automate it for free:** push this folder to a GitHub repo, add the key as a
repo secret named `DATAGOV_API_KEY`, and `.github/workflows/daily-data.yml`
saves the prices and the weather forecast every evening at 20:45 IST, even
with your laptop off. Run `git pull` before working so you have the new files.
Run `python -m krishibhav load-datagov` to load all saved days into your database.

## 4. Weather

```bat
python -m krishibhav fetch-weather
```

Pulls daily rainfall, temperature and evapotranspiration since 2014 for Deesa,
Palanpur, Dhanera and Unjha, plus a 16-day forecast (Open-Meteo, no key needed).
Re-running fetches the new days plus the last 10 (Open-Meteo corrects recent
days). Every forecast is kept in `weather_forecast` with its issue date, so
back-tests can use what was forecast at the time. Each forecast is first saved
as `data/raw/weather/forecast/<date>.json`; the GitHub job saves one every
evening (old forecasts cannot be downloaded later, so these files are the
record). After `git pull`, run `python -m krishibhav load-weather` to load them. Coordinates are in `krishibhav/config.py`;
check them on a map and add growing areas if you want.

## 5. Build the modelling dataset

```bat
python -m krishibhav build-series
```

Writes `data/processed/daily_series.csv`: one row per day per market and
commodity, using one variety per series (the one reported on the most days, or
the one you set in `MAIN_VARIETY` in `config.py`; merge spellings with
`VARIETY_ALIASES`). Several grades on the same day are combined into one price
(weighted by arrivals). Prices more than 3x away from the surrounding days are
treated as typos and blanked. Gaps of up to 3 days (Sundays, holidays) are
filled and marked `filled=True`; longer gaps stay empty.

## Commands

| Command | What it does |
| --- | --- |
| `init-db` | Create the database tables |
| `load-files [--keep-all]` | Clean and load everything in `data/raw/agmarknet` |
| `fetch-today [--raw-only]` | Today's Gujarat prices from data.gov.in |
| `load-datagov` | Re-load all saved daily API responses |
| `fetch-weather [--forecast-only] [--raw-only]` | Weather history and forecast |
| `load-weather` | Re-load all saved forecast files |
| `build-series` | Write `data/processed/daily_series.csv` |
| `report` | Coverage table and charts in `reports/` |
| `daily` | `fetch-today` + `fetch-weather` + `build-series` |

## Project layout

```
krishibhav/
  config.py     commodities, markets, coordinates  ← edit this first
  clean.py      reads messy exports, fixes names, numbers and dates
  db.py         tables + upsert (re-loading the same file never duplicates rows)
  ingest.py     loads downloaded files and saved API responses
  datagov.py    data.gov.in daily fetcher with retries
  weather.py    Open-Meteo history + forecast
  series.py     one variety, one clean daily series per market and commodity
  report.py     coverage table and charts
tests/          50 tests on fake Agmarknet/API data
```

## If something breaks

- **"missing columns ['date', 'modal_price']"**: the export has an unusual
  layout. Open it, check the header names, and send them to me; usually it's a
  one-line fix in `canon_col()` in `clean.py`.
- **data.gov.in returns nothing**: the yard hasn't reported yet (try later), the
  site is down (it often is), or field names changed. Check the dataset page.
- **Switch to PostgreSQL** later: install `psycopg2-binary` and set
  `DATABASE_URL` in `.env`, then run `load-files` and `load-datagov` again.

Note: the downloaders were tested against the documented API formats with
fake responses, not live calls. Expect a small fix or two on the first real run.
