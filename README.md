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
On Mac/Linux use `source .venv/bin/activate` and `cp`. All 98 tests should pass.
They use fake files, so no internet is needed.

## 2a. First check which markets report enough

```bat
python -m krishibhav check-markets --markets "Tharad,Diyodar"
```

Uses data.gov.in's Agmarknet price history (needs the API key from step 3) to
show, per market and crop, the share of trading days (Mon-Sat) with a price
in each year since 2014, and the main variety. Extra markets are optional;
use their exact Agmarknet names. Results go to `reports/market_check.csv`.

The downloaded rows are kept in `data/raw/agmarknet/datagov_hist/`, so
`load-files` also loads them as a ready-made **price** backfill. They have no
arrivals, so you still need the Agmarknet or CEDA reports below for arrivals.

## 2b. Download historical prices and arrivals (one evening of clicking)

1. Open **agmarknet.gov.in → Price and Arrivals** (commodity-wise or market-wise report).
2. Choose State **Gujarat**, District **Banaskantha**, Market **Deesa** (also try
   **Deesa Bhildi** and **Deesa Veg Yard** for potato), the commodity, and a date range.
   If the site refuses long ranges, download one year at a time.
3. Export to Excel and save into `data/raw/agmarknet/` with a name like
   `potato_deesa-veg-yard_2015.xlsx` (see the README in that folder).
4. Repeat for each commodity (potato, castor, groundnut, bajra) and
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

Each day's raw response is saved as `data/raw/datagov/<date>.json`, named by
the date the prices are for (the API only gives the latest day, so these files
become your live history).

**Automate it on your laptop.** data.gov.in does not answer GitHub's servers
(they are outside India), so prices are fetched by a Windows scheduled task.
Set it up once:

```bat
powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
```

It runs `scripts\daily_prices.ps1` at 21:00 and 23:30 (or as soon as the laptop
is back on): pull, `fetch-today`, commit only `data/raw/datagov`, push. It
only commits on `main` and does not push if you have your own unpushed
commits. Check `logs\daily_prices.log` to see what it did. If you miss days,
`check-markets --refresh` gets them back from the Agmarknet history later.

The GitHub job (`.github/workflows/daily-data.yml`) saves only the weather
forecast, every evening at 20:45 IST. Run `git pull` before working so you have
the new files, and `python -m krishibhav load-datagov` to load all saved
price days into your database.

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
commodity. Deesa, Deesa Bhildi and Deesa Veg Yard are combined into one
**Deesa** series (`MARKET_GROUPS` in `config.py`), because each yard alone
reports on only about half the trading days. Each yard uses one variety (the one reported on the most days, or
the one you set in `MAIN_VARIETY` in `config.py`; merge spellings with
`VARIETY_ALIASES`). Several grades on the same day are combined into one price
(weighted by arrivals). Prices more than 3x away from the surrounding days are
treated as typos and blanked. Gaps of up to 3 days (Sundays, holidays) are
filled and marked `filled=True`; longer gaps stay empty.

## 6. Reference tables (MSP, calendar, crop seasons)

Hand-made CSVs in `data/reference/`, committed to git. Edit them in Excel or a
text editor; lines starting with `#` are comments.

| File | What it holds |
| --- | --- |
| `msp.csv` | MSP per crop and season (groundnut, bajra; castor and potato have none). Season `2025-26` applies 1 Oct 2025 to 30 Sep 2026. Add each new season in May-June. |
| `calendar.csv` | Dated events: `festival` (Diwali, Holi, Uttarayan), `apmc_holiday`, `policy` (export bans, duty changes). Leave `crop` / `market` empty for "all". |
| `crop_seasons.csv` | Sowing and harvest windows that repeat every year (MM-DD, may cross 31 Dec). |

Rows marked `status=check` still need confirming (with the shop, or the
CACP/PIB release for MSP). After editing, run:

```bat
python -m krishibhav check-reference
```

It stops with a list of every typo it finds, and lists Mon-Sat days when no
yard reported anything (likely holidays) in `reports/closed_day_candidates.csv`,
so you can add the real APMC holidays. In Python,
`reference.daily_flags(dates, "Potato", market="Deesa")` gives the daily
features: festival / apmc_holiday / policy flags, days to Diwali, `in_harvest`
etc., and the MSP in force.

## 7. Feature table (M2)

```bat
python -m krishibhav build-features
```

Writes `data/processed/features.csv`: one row per day for each Deesa crop
(`FEATURE_MARKETS` in `config.py`), with every feature built only from data
known on that day, and targets `y_ret_7/14/30` (log return to 7/14/30 days
ahead) plus `y_price_7/14/30` (the price then, in Rs/quintal).

| Group | Columns |
| --- | --- |
| Price | `log_price`, `ret_1/7/14/30/365`, `ma_gap_7/30/90`, `vol_7/30/90`, `spread`, `reported`, `days_since_report` |
| Arrivals | `arr`, `arr_7`, `arr_ratio_ly` (empty until an arrivals source is loaded) |
| Weather | `rain_7/30/90`, `tmean_30`, `tmax_7`, `et0_30` |
| Policy | `msp_rs_qtl`, `price_to_msp` (groundnut and bajra only) |
| Calendar | `dow`, `month`, `week`, `doy_sin/cos`, `festival`, `apmc_holiday`, `policy`, `days_to_diwali`, `in_harvest` etc. |
| Lead markets | `x_palanpur_gap`, `x_palanpur_ret_7`, same for Dhanera |
| Seasonal | `seas_ret_7/14/30`: last year's change over the same window |

The `split` column follows the plan, in time order only: `train` 2015-2022,
`valid` 2023, `test` 2024-2025 (`warmup` 2014, `live` after 2025). Rows include
days filled from the previous report (`reported=False`); evaluate on reported
days. A test checks that changing anything after day t never changes a feature
on or before t. Column coverage is saved in `reports/feature_coverage.csv`.

## 8. Baselines and XGBoost (M3)

```bat
python -m krishibhav evaluate
```

Compares three models on every horizon, walk-forward over 2023 (valid) and
2024-2025 (test): each month XGBoost is retrained only on rows whose outcome
was already known before that month, then predicts the month.

| Model | Forecast |
| --- | --- |
| `naive` | price in h days = today's price |
| `seasonal` | today's price x last year's change over the same window |
| `xgb` | one XGBoost model for all crops, quantile loss: 10th / 50th / 90th percentile |

Scores (on days with a real report): MAPE, RMSE (Rs/quintal), directional
accuracy (did it call up vs down) and, for XGBoost, how often the actual price
fell inside the 10-90 range (about 80% is honest).

- **Explanations:** every XGBoost forecast lists its top 3 drivers (SHAP values,
  related features summed into plain labels such as "time of year" or "price vs
  MSP") with their effect in % price change: `driver_1..3`, `effect_1..3`.
- **Calibrated ranges:** the 10-90 range is shifted per crop and horizon so that
  about 10% of outcomes fall below and 10% above, learned only from the previous
  12 months of errors that were already known (`q10`, `q90`; the model's own are
  `q10_raw`, `q90_raw`). The walk-forward starts 12 months early for this; those
  months are not scored.
- **Diebold-Mariano test:** is XGBoost's absolute % error significantly lower
  than naive's, per crop and horizon (Newey-West variance, Harvey-Leybourne-
  Newbold correction).

Results go to `reports/m3_results.md` (tables for the report, best value in
bold, coverage, DM test, top drivers), `reports/m3_metrics.csv`,
`reports/m3_dm_test.csv` and `reports/m3_shap_importance.csv`; every prediction
to `data/processed/predictions_m3.csv`. Options: `--models naive,xgb`,
`--horizons 7`. A full run takes several minutes.

## 9. Sell-or-hold advisor (M5)

```bat
python -m krishibhav backtest
```

For each day and horizon h the advisor computes the expected net gain of
holding one quintal, `G_h = P50(t+h) - P(t) - c_s*h - P(t)*r*h/365`, and with
the best h: **hold** if `G_h` beats the threshold and the 10th-percentile price
is above today's; **sell half** if it beats the threshold but the 10th
percentile is lower; otherwise **sell now**. Storage cost `c_s` per crop,
interest `r` and the threshold are in `config.py`
(`STORAGE_COST_RS_QTL_DAY`, `INTEREST_RATE`, `HOLD_THRESHOLD`).

The back-test replays every reported day of 2023-2025 using the walk-forward
forecasts from `evaluate` (run that first), follows the advice for one quintal
and compares the money received, after storage and interest, with selling on
arrival. The threshold is chosen on 2023 only. It also reports "always hold 30
days", perfect foresight (the upper bound) and a block-bootstrap 95% interval
for the gain. Output: `reports/m5_backtest.md` (with example advice cards in
Gujarati and English), `reports/m5_backtest.csv`,
`data/processed/advisor_backtest.csv`.

`advisor.advice_text(row, lot_qtl)` gives the advice, price range, expected
gain for the lot and top 3 reasons in Gujarati and English. **Check the
Gujarati wording with your father and a few farmers.**

## 10. Today's advice (for the web app)

```bat
python -m krishibhav advise --lot 50
```

Rebuilds the feature table from the database (add `--no-rebuild` to skip),
trains XGBoost on every row whose outcome is already known (saved in
`models/`, not in git), and forecasts each Deesa crop from its latest reported
price. The 10-90 range is calibrated on that crop's most recent 12 months of
walk-forward errors, so run `evaluate` every month or so to keep them recent.
The advisor rules then give the advice, printed as Gujarati / English cards
and written to **`data/processed/advice_latest.json`**, the file the web app
reads: per crop the action, best horizon, price range for 7/14/30 days,
expected gain per quintal and for the lot, top 3 reasons (English and
Gujarati) and the disclaimer.

If a crop's latest Deesa price is more than 7 days old the card warns about
it; after 30 days it gives no advice ("no recent price") instead of guessing.
`--crop Potato` limits it to one crop.

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
| `build-features` | Write `data/processed/features.csv` (model inputs + targets) |
| `evaluate [--models ...] [--horizons ...]` | M3 walk-forward comparison, results in `reports/m3_results.md` |
| `backtest [--lot 50]` | M5 advisor revenue back-test, results in `reports/m5_backtest.md` |
| `advise [--lot 50] [--crop ...] [--no-rebuild]` | Today's advice per Deesa crop -> `data/processed/advice_latest.json` |
| `report` | Coverage table and charts in `reports/` |
| `check-markets [--markets "A,B"] [--refresh]` | Reporting days per market, crop and year |
| `check-reference` | Validate `data/reference/*.csv`, suggest APMC holidays |
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
  markets.py    check-markets: how often each market reports (Agmarknet history)
  reference.py  MSP, calendar and crop-season tables -> daily flags
  features.py   M2 feature table: lags, rolling stats, weather, MSP, calendar, targets
  models.py     M3 baselines + XGBoost, walk-forward, SHAP drivers, calibrated ranges, DM test
  advisor.py    M5 sell / hold rules, revenue back-test, Gujarati + English advice text
  live.py       advise: train on all data, today's advice per crop as JSON for the app
tests/          98 tests on fake Agmarknet/API data and the reference CSVs
scripts/        daily_prices.ps1 (laptop price job) + install_task.ps1 (schedules it)
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
