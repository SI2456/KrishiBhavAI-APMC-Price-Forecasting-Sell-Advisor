"""Tests for the data pipeline. Run: pytest -q

They use small fake files shaped like real Agmarknet / data.gov.in exports,
so no internet or API key is needed.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select

from krishibhav import clean, config, datagov, db, ingest, report, series, weather

HEADER = ["Sl no.", "District Name", "Market Name", "Commodity", "Variety", "Grade",
          "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)", "Modal Price (Rs./Quintal)", "Price Date"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(config, "RAW_AGMARKNET", tmp_path / "raw" / "agmarknet")
    monkeypatch.setattr(config, "RAW_DATAGOV", tmp_path / "raw" / "datagov")
    monkeypatch.setattr(config, "RAW_WEATHER_FORECAST", tmp_path / "raw" / "weather" / "forecast")
    monkeypatch.setattr(config, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    config.RAW_AGMARKNET.mkdir(parents=True)
    config.RAW_DATAGOV.mkdir(parents=True)
    return tmp_path


def agmarknet_xlsx(path):
    """Two title rows above the header, like the portal's Excel export."""
    rows = [
        ["Commodity wise price report, Gujarat"] + [""] * 9,
        [""] * 10,
        HEADER,
        [1, "Banaskantha", "Deesa(Deesa Veg Yard)", "Potato", "Desi", "FAQ", "1,000", "1,400", "1,200", "17 Oct 2026"],
        [2, "Banaskantha", "Deesa(Deesa Veg Yard)", "Potato", "Jyoti", "FAQ", "900", "1,300", "1,100", "17 Oct 2026"],
        [3, "Banaskantha", "Deesa(Deesa Veg Yard)", "Potato", "Desi", "FAQ", "0", "0", "0", "16 Oct 2026"],
        [4, "Rajkot", "Rajkot", "Potato", "Desi", "FAQ", "800", "1,000", "900", "17 Oct 2026"],
        [5, "Banaskantha", "Deesa", "Onion", "Red", "FAQ", "800", "1,000", "900", "17 Oct 2026"],
    ]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


# ------------------------------------------------------------------ cleaning
@pytest.mark.parametrize("raw, expected", [
    ("Modal Price (Rs./Quintal)", "modal_price"),
    ("Min Price (Rs./Quintal)", "min_price"),
    ("max_price", "max_price"),
    ("Arrival_Date", "date"),
    ("Price Date", "date"),
    ("Arrivals (Tonnes)", "arrivals_tonnes"),
    ("arrivals_unit", None),
    ("Market Name", "market"),
    ("market_center", "market"),
    ("market_code", None),
    ("Commodity", "commodity"),
    ("commodity_type", None),
    ("Sl no.", None),
    ("state_id", None),
    ("Market ID", None),
    ("commodity_id", None),
    ("market_name", "market"),
])
def test_canon_col(raw, expected):
    assert clean.canon_col(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Deesa(Deesa Veg Yard)", "Deesa Veg Yard"),
    ("DEESA (BHILDI)", "Deesa Bhildi"),
    ("Deesa APMC", "Deesa"),
    ("palanpur", "Palanpur"),
])
def test_norm_market(raw, expected):
    assert clean.norm_market(raw) == expected


def test_norm_commodity():
    assert clean.norm_commodity("Potato(Desi)") == "Potato"
    assert clean.norm_commodity("Castor Seed") == "Castor Seed"
    assert clean.norm_commodity("Onion") is None


@pytest.mark.parametrize("raw, expected", [
    ("Sweet Potato", None),
    ("Amaranthus", None),                # leafy vegetable, not rajgira grain
    ("Groundnut Oil", None),
    ("Groundnut Seed", None),            # kernels
    ("Groundnut (Split)", None),
    ("Castor Oil", None),
    ("Groundnut pods (raw)", "Groundnut"),
    ("Rajgir", "Rajgira"),
    ("Rajgira", "Rajgira"),
    ("Cumin Seed(Jeera)", "Cumin"),
    ("Bajra(Pearl Millet/Cumbu)", "Bajra"),
])
def test_norm_commodity_other_products(raw, expected):
    assert clean.norm_commodity(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("2020-02-01", date(2020, 2, 1)),            # ISO: never day first
    ("2026-10-05 00:00:00", date(2026, 10, 5)),  # Excel date cell read as text
    ("2020/2/1", date(2020, 2, 1)),
    ("05/10/2026", date(2026, 10, 5)),           # Indian day first
    ("05-10-2026", date(2026, 10, 5)),
    ("17 Oct 2026", date(2026, 10, 17)),
])
def test_to_date(raw, expected):
    assert clean.to_date(pd.Series([raw])).iloc[0] == expected


def test_to_date_bad_values():
    out = clean.to_date(pd.Series(["", None, "junk", "2020-13-01"]))
    assert out.isna().all()


def test_xlsx_with_real_date_cells(env):
    path = config.RAW_AGMARKNET / "castor_deesa_2026.xlsx"
    pd.DataFrame({"Market Name": ["Deesa", "Deesa"], "Commodity": ["Castor Seed"] * 2,
                  "Modal Price": [6300, 6350],
                  "Price Date": [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")]}
                 ).to_excel(path, index=False)
    df, _ = clean.normalise(clean.read_table(path), source="agmarknet_file", file_hint=path.stem)
    assert df["date"].tolist() == [date(2026, 10, 5), date(2026, 10, 6)]


def test_ceda_style_id_columns(env):
    """ID columns come before name columns; the names must win."""
    df = pd.DataFrame([["2020-02-01", "24", "Gujarat", "445", "Banaskantha", "1001", "Deesa",
                        "23", "Potato", "Desi", "900", "1100", "1000", "50"]],
                      columns=["date", "state_id", "state_name", "district_id", "district_name",
                               "market_id", "market_name", "commodity_id", "commodity_name",
                               "variety", "min_price", "max_price", "modal_price", "arrivals"])
    out, _ = clean.normalise(df, source="ceda_file", file_hint="ceda_potato")
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["market"], row["commodity"], row["district"]) == ("Deesa", "Potato", "Banaskantha")
    assert row["date"] == date(2020, 2, 1)


def test_min_max_swapped():
    df = pd.DataFrame({"Market": ["Deesa"], "Commodity": ["Castor Seed"], "Price Date": ["17/10/2026"],
                       "Min Price": ["6,450"], "Max Price": ["6,100"], "Modal Price": ["6,300"]})
    out, rep = clean.normalise(df, source="agmarknet_file")
    assert (out.loc[0, "min_price"], out.loc[0, "max_price"]) == (6100.0, 6450.0)
    assert rep.min_max_swapped == 1 and rep.modal_outside_range == 0


def test_xlsx_with_title_rows(env):
    path = config.RAW_AGMARKNET / "potato_deesa_2026.xlsx"
    agmarknet_xlsx(path)
    df, rep = clean.normalise(clean.read_table(path), source="agmarknet_file", file_hint=path.stem)
    assert len(df) == 2                                   # zero-price, Rajkot and onion rows dropped
    assert set(df["market"]) == {"Deesa Veg Yard"}
    assert df["modal_price"].tolist() == [1200.0, 1100.0]
    assert df["date"].iloc[0] == date(2026, 10, 17)
    assert rep.dropped["no price or arrivals"] == 1


def test_html_disguised_as_xls(env):
    table = pd.DataFrame([[1, "Banaskantha", "Deesa", "Castor Seed", "Castor", "FAQ",
                           "6,100", "6,450", "6,385", "17/10/2026"]], columns=HEADER)
    path = config.RAW_AGMARKNET / "castor_report.xls"
    path.write_text(table.to_html(index=False), encoding="utf-8")
    df, _ = clean.normalise(clean.read_table(path), source="agmarknet_file", file_hint=path.stem)
    assert df.loc[0, "commodity"] == "Castor Seed"
    assert df.loc[0, "modal_price"] == 6385.0


def test_commodity_and_market_from_file_name(env):
    path = config.RAW_AGMARKNET / "groundnut_deesa-bhildi_2020.csv"
    pd.DataFrame({"Price Date": ["01/02/2020", "03/02/2020"], "Variety": ["Bold", "Bold"],
                  "Modal Price": ["5,500", "5,600"]}).to_csv(path, index=False)
    df, _ = clean.normalise(clean.read_table(path), source="agmarknet_file", file_hint=path.stem)
    assert set(df["commodity"]) == {"Groundnut"}
    assert set(df["market"]) == {"Deesa Bhildi"}


# ------------------------------------------------------------------ database
def test_load_files_and_coalesce_upsert(env):
    agmarknet_xlsx(config.RAW_AGMARKNET / "potato_deesa_2026.xlsx")
    # A separate arrivals file for the same keys (prices left blank)
    pd.DataFrame({
        "Market Name": ["Deesa(Deesa Veg Yard)"], "Commodity": ["Potato"], "Variety": ["Desi"],
        "Grade": ["FAQ"], "Arrivals (Tonnes)": ["350"], "Modal Price": ["1200"], "Price Date": ["17-10-2026"],
    }).to_csv(config.RAW_AGMARKNET / "zz_potato_arrivals.csv", index=False)

    assert ingest.load_files() == 3
    engine = db.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(select(db.mandi_prices)).mappings().all()
    assert len(rows) == 2                                 # same key merged, not duplicated
    desi = next(r for r in rows if r["variety"] == "Desi")
    assert desi["arrivals_tonnes"] == 350.0
    assert desi["min_price"] == 1000.0                    # not blanked by the arrivals file

    assert ingest.load_files() == 3                       # re-running is safe
    with engine.connect() as conn:
        assert len(conn.execute(select(db.mandi_prices)).all()) == 2


def test_arrivals_only_file_merges(env):
    """An arrivals report with no price columns at all still loads and merges."""
    agmarknet_xlsx(config.RAW_AGMARKNET / "potato_deesa_2026.xlsx")
    pd.DataFrame({
        "Market Name": ["Deesa(Deesa Veg Yard)"], "Commodity": ["Potato"], "Variety": ["Desi"],
        "Grade": ["FAQ"], "Arrivals (Tonnes)": ["350"], "Arrival Date": ["17/10/2026"],
    }).to_csv(config.RAW_AGMARKNET / "zz_potato_arrivals_only.csv", index=False)

    assert ingest.load_files() == 3
    with db.get_engine().connect() as conn:
        rows = conn.execute(select(db.mandi_prices)).mappings().all()
    assert len(rows) == 2
    desi = next(r for r in rows if r["variety"] == "Desi")
    assert (desi["arrivals_tonnes"], desi["modal_price"]) == (350.0, 1200.0)


def test_series_fills_short_gaps_only(env):
    rows = []
    for d, price in [("2026-01-01", 100), ("2026-01-03", 110), ("2026-01-10", 120)]:
        rows.append({"date": pd.Timestamp(d), "market": "Deesa", "commodity": "Castor Seed",
                     "modal_price": price, "min_price": price - 5, "max_price": price + 5,
                     "arrivals_tonnes": None})
    out = series.fill_calendar(series.combine_daily(pd.DataFrame(rows)), max_fill=3)
    by_day = out.set_index("date")
    assert by_day.loc["2026-01-02", "modal_price"] == 100 and by_day.loc["2026-01-02", "filled"]
    assert by_day.loc["2026-01-06", "modal_price"] == 110      # 3 days after Jan 3
    assert pd.isna(by_day.loc["2026-01-07", "modal_price"])    # gap too long, left empty
    assert not by_day.loc["2026-01-10", "filled"]


def test_arrival_weighted_price():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01")] * 2, "market": ["Deesa"] * 2, "commodity": ["Potato"] * 2,
        "modal_price": [1000.0, 1300.0], "min_price": [900.0, 1200.0], "max_price": [1100.0, 1400.0],
        "arrivals_tonnes": [300.0, 100.0],
    })
    out = series.combine_daily(df)
    assert out.loc[0, "modal_price"] == 1075.0            # (1000*300 + 1300*100) / 400


def price_rows(days, market="Deesa", commodity="Potato", variety="Desi", start="2026-01-01", price=1000.0):
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"date": dates, "market": market, "commodity": commodity, "variety": variety,
                         "modal_price": price, "min_price": price - 50, "max_price": price + 50,
                         "arrivals_tonnes": None})


def test_pick_variety_most_days_and_override(monkeypatch):
    df = pd.concat([price_rows(10, variety="Desi"), price_rows(4, variety="Jyoti", price=1500.0)])
    assert set(series.pick_variety(df)["variety"]) == {"Desi"}            # most reported days

    monkeypatch.setattr(config, "MAIN_VARIETY", {"Potato": "Jyoti"})
    assert set(series.pick_variety(df)["variety"]) == {"Jyoti"}           # chosen by hand

    monkeypatch.setattr(config, "MAIN_VARIETY", {"Potato": "Chipsona"})   # not in data: fall back
    assert set(series.pick_variety(df)["variety"]) == {"Desi"}


def test_variety_aliases_merge_spellings(monkeypatch):
    monkeypatch.setattr(config, "VARIETY_ALIASES", {"Potato": {"Desi": ["desi", "local"]}})
    df = pd.concat([price_rows(3, variety="Local"), price_rows(3, variety="Desi", start="2026-01-04"),
                    price_rows(5, variety="Jyoti")])
    out = series.pick_variety(df)
    assert set(out["variety"]) == {"Desi"} and len(out) == 6               # 6 Desi days beat 5 Jyoti


def test_arrivals_only_row_does_not_dilute_price():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01")] * 2, "market": ["Deesa"] * 2, "commodity": ["Potato"] * 2,
        "modal_price": [1000.0, None], "min_price": [900.0, None], "max_price": [1100.0, None],
        "arrivals_tonnes": [100.0, 300.0],
    })
    out = series.combine_daily(df)
    assert out.loc[0, "modal_price"] == 1000.0
    assert out.loc[0, "arrivals_tonnes"] == 400.0


def test_drop_spikes():
    daily = series.combine_daily(price_rows(21))
    daily.loc[10, "modal_price"] = 100000.0                                # Rs/kg typed wrong
    daily.loc[5, "modal_price"] = 1300.0                                   # a real 30% move stays
    out = series.drop_spikes(daily).set_index("date")
    assert pd.isna(out.loc["2026-01-11", "modal_price"]) and pd.isna(out.loc["2026-01-11", "max_price"])
    assert out.loc["2026-01-06", "modal_price"] == 1300.0
    assert out["modal_price"].notna().sum() == 20


def test_coverage_counts_whole_history_window():
    full = price_rows(365, start="2025-01-01")
    late = price_rows(30, market="Palanpur", start="2025-12-02")
    daily = series.combine_daily(pd.concat([full, late]))
    table = report.coverage_table(daily, start="2025-01-01").set_index("market")
    assert table.loc["Deesa", "coverage_pct"] == 100.0
    assert table.loc["Palanpur", "coverage_pct"] < 10                     # used to show 100%
    assert table.loc["Palanpur", "longest_gap_days"] == 335               # Jan 1 to Dec 1 empty
    assert table.loc["Deesa", "longest_gap_days"] == 0


def test_weather_refetches_recent_days_and_keeps_forecasts(env, monkeypatch):
    class FakeDate(date):
        current = date(2026, 10, 20)

        @classmethod
        def today(cls):
            return cls.current

    calls = []

    def fake_call(url, params):
        calls.append((url, params))
        if url == weather.ARCHIVE:
            days = pd.date_range(params["start_date"], params["end_date"], freq="D")
        else:
            days = pd.date_range(FakeDate.today(), periods=3, freq="D")
        return {"daily": {"time": [f"{d:%Y-%m-%d}" for d in days], "precipitation_sum": [1.0] * len(days)}}

    monkeypatch.setattr(weather, "date", FakeDate)
    monkeypatch.setattr(weather, "_call", fake_call)
    monkeypatch.setattr(config, "WEATHER_LOCATIONS", {"Deesa": (24.26, 72.19)})
    monkeypatch.setattr(config, "HISTORY_START", "2026-10-01")

    weather.fetch_weather()
    FakeDate.current = date(2026, 10, 21)
    weather.fetch_weather()

    starts = [p["start_date"] for u, p in calls if u == weather.ARCHIVE]
    assert starts == ["2026-10-01", "2026-10-05"]                          # last 10 days fetched again
    with db.get_engine().connect() as conn:
        history = conn.execute(select(db.weather_daily)).mappings().all()
        forecasts = conn.execute(select(db.weather_forecast)).mappings().all()
    assert len(history) == 15                                              # Oct 1 to 15, no duplicates
    assert len(forecasts) == 6                                             # 2 issues x 3 days, both kept
    assert {str(r["issued"]) for r in forecasts} == {"2026-10-20", "2026-10-21"}

    saved = sorted(p.name for p in config.RAW_WEATHER_FORECAST.glob("*.json"))
    assert saved == ["2026-10-20.json", "2026-10-21.json"]                 # raw files kept too


def test_forecast_raw_only_then_load(env, monkeypatch):
    """The GitHub job only saves JSON; load-weather later puts it in the database."""
    def fake_call(url, params):
        assert url == weather.FORECAST                                     # no history download
        return {"daily": {"time": ["2026-10-20", "2026-10-21"], "precipitation_sum": [0.0, 4.2]}}

    monkeypatch.setattr(weather, "_call", fake_call)
    monkeypatch.setattr(config, "WEATHER_LOCATIONS", {"Deesa": (24.26, 72.19), "Unjha": (23.80, 72.39)})

    assert weather.fetch_weather(load=False) == 0
    assert not (env / "test.db").exists()                                  # database untouched
    (path,) = config.RAW_WEATHER_FORECAST.glob("*.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved["locations"]) == {"Deesa", "Unjha"}

    assert weather.load_forecasts_raw() == 4
    assert weather.load_forecasts_raw() == 4                               # re-loading is safe
    with db.get_engine().connect() as conn:
        rows = conn.execute(select(db.weather_forecast)).mappings().all()
    assert len(rows) == 4 and {str(r["issued"]) for r in rows} == {saved["issued"]}


# ------------------------------------------------------------------ APIs (no network)
def test_datagov_pagination_and_filter_fallback(env, monkeypatch):
    calls = []
    record = {"state": "Gujarat", "district": "Banaskantha", "market": "Deesa", "commodity": "Castor Seed",
              "variety": "Castor", "grade": "FAQ", "arrival_date": "17/10/2026",
              "min_price": "6100", "max_price": "6450", "modal_price": "6385"}

    def fake_get(session, url, params, tries=4):
        assert "python-requests" not in session.headers["User-Agent"]   # data.gov.in stalls on it
        calls.append(params)
        if "filters[state.keyword]" in params:          # first spelling returns nothing
            return {"records": [], "total": 0}
        if params["offset"] == 0:
            return {"records": [record] * 1000, "total": 1001}
        return {"records": [dict(record, market="Palanpur")], "total": 1001}

    monkeypatch.setattr(datagov, "_get", fake_get)
    monkeypatch.setattr(config, "DATAGOV_API_KEY", "test-key")
    assert len(datagov.fetch_state("test-key")) == 1001
    assert [c["offset"] for c in calls if "filters[state]" in c] == [0, 1000]

    n = datagov.fetch_today()
    assert n == 2                                          # 1000 identical rows dedupe to one + Palanpur
    saved = list(config.RAW_DATAGOV.glob("*.json"))
    assert len(saved) == 1 and len(json.loads(saved[0].read_text())["records"]) == 1001
    assert ingest.load_datagov_raw() == 2


def test_weather_rows():
    payload = {"daily": {"time": ["2026-10-01", "2026-10-02"],
                         "precipitation_sum": [0.0, 12.5], "temperature_2m_max": [34.1, 31.0]}}
    rows = weather._rows(payload, "Deesa", kind="history")
    assert rows[1]["precipitation_sum"] == 12.5 and rows[1]["kind"] == "history"
    assert rows[1]["temperature_2m_min"] is None
    assert rows[0]["date"] == date(2026, 10, 1)


def test_datagov_skips_filter_the_api_ignored(monkeypatch):
    gujarat = {"state": "Gujarat", "market": "Deesa", "commodity": "Potato", "modal_price": "1200"}
    punjab = dict(gujarat, state="Punjab")
    calls = []

    def fake_get(session, url, params, tries=4):
        calls.append(params)
        if "filters[state.keyword]" in params:          # filter ignored: all of India comes back
            return {"records": [gujarat, punjab], "total": 50000}
        return {"records": [gujarat], "total": 1}

    monkeypatch.setattr(datagov, "_get", fake_get)
    assert datagov.fetch_state("test-key") == [gujarat]
    assert len(calls) == 2                               # did not page through all of India
