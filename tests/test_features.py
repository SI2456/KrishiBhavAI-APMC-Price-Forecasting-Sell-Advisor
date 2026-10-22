"""Tests for the M2 feature table: values, targets, splits and no look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krishibhav import config, features, reference, series


@pytest.fixture
def refs(tmp_path):
    (tmp_path / "msp.csv").write_text("""crop,season,msp_rs_qtl,status,source
Bajra,2014-15,1250,check,
Bajra,2015-16,1275,verified,
Bajra,2016-17,1330,verified,
""", encoding="utf-8")
    (tmp_path / "calendar.csv").write_text("""start,end,kind,name,crop,market,status
2015-11-11,,festival,Diwali,,,verified
2016-10-30,,festival,Diwali,,,verified
2016-10-28,2016-11-04,apmc_holiday,Diwali vacation,,,check
""", encoding="utf-8")
    (tmp_path / "crop_seasons.csv").write_text("""crop,phase,start,end,status,note
Potato,harvest,01-15,03-31,check,
Potato,cold_storage_release,04-01,11-30,check,
Bajra,harvest,09-15,10-31,check,
""", encoding="utf-8")
    return {"msp": reference.load_msp(tmp_path / "msp.csv"),
            "calendar": reference.load_calendar(tmp_path / "calendar.csv"),
            "seasons": reference.load_seasons(tmp_path / "crop_seasons.csv")}


def make_daily(prices: dict[tuple[str, str], pd.Series]) -> pd.DataFrame:
    """{(market, commodity): price series indexed by date} -> fill_calendar output."""
    rows = []
    for (market, commodity), p in prices.items():
        p = p.dropna()
        rows.append(pd.DataFrame({"date": p.index, "market": market, "commodity": commodity,
                                  "variety": "Other", "modal_price": p.to_numpy(),
                                  "min_price": p.to_numpy() * 0.9, "max_price": p.to_numpy() * 1.1,
                                  "arrivals_tonnes": np.nan}))
    return series.fill_calendar(series.combine_daily(pd.concat(rows, ignore_index=True)))


def make_weather(days: pd.DatetimeIndex, rain: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"location": "Deesa", "date": days, "precipitation_sum": rain,
                         "temperature_2m_max": 35.0, "temperature_2m_min": 20.0,
                         "temperature_2m_mean": 27.0, "et0_fao_evapotranspiration": 5.0})


def random_prices(days, seed, level):
    rng = np.random.default_rng(seed)
    p = pd.Series(level * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))), index=days)
    p[rng.random(len(days)) < 0.3] = np.nan                     # ~30% of days not reported
    return p


def test_no_look_ahead(refs):
    """Changing prices and weather after day T must not change any feature up to T."""
    days = pd.date_range("2015-01-01", "2016-12-31")
    cut = pd.Timestamp("2016-06-30")
    base = {("Deesa", "Potato"): random_prices(days, 1, 1000),
            ("Palanpur", "Potato"): random_prices(days, 2, 950),
            ("Deesa", "Bajra"): random_prices(days, 3, 1300)}
    shocked = {k: v.where(v.index <= cut, v * 3) for k, v in base.items()}
    weather = make_weather(days)
    weather_shocked = weather.assign(precipitation_sum=np.where(weather["date"] > cut, 50.0, 1.0))

    a = features.build_features(make_daily(base), weather, refs)
    b = features.build_features(make_daily(shocked), weather_shocked, refs)
    key = ["date", "market", "commodity"]
    a, b = a[a["date"] <= cut].set_index(key), b[b["date"] <= cut].set_index(key)
    feature_cols = [c for c in a.columns if not c.startswith("y_")]
    pd.testing.assert_frame_equal(a[feature_cols], b.loc[a.index, feature_cols])
    # ...while targets that reach past T do change, so the shock was real
    assert not np.allclose(a["y_ret_30"].dropna(), b.loc[a["y_ret_30"].dropna().index, "y_ret_30"])


def test_price_features_and_targets(refs):
    days = pd.date_range("2015-01-01", "2015-03-31")
    p = pd.Series(1000.0 * 1.01 ** np.arange(len(days)), index=days)   # +1% a day
    p.loc["2015-02-10":"2015-02-11"] = np.nan                           # two-day gap, filled
    t = features.build_features(make_daily({("Deesa", "Potato"): p}), make_weather(days), refs)
    t = t.set_index("date")

    row = t.loc["2015-02-01"]
    assert row["ret_7"] == pytest.approx(7 * np.log(1.01), rel=1e-3)        # prices are rounded to 0.01
    assert row["y_ret_7"] == pytest.approx(7 * np.log(1.01), rel=1e-3)
    assert row["y_price_7"] == pytest.approx(p["2015-02-08"], abs=0.01)
    assert row["spread"] == pytest.approx(0.2, rel=1e-4)                         # (1.1 - 0.9) / 1
    assert row["rain_7"] == 7.0 and np.isnan(row["rain_90"])            # 90 days not available yet
    assert t.loc["2015-02-11", "days_since_report"] == 2 and not t.loc["2015-02-11", "reported"]
    assert t.loc["2015-02-11", "log_price"] == pytest.approx(np.log(p["2015-02-09"]), abs=1e-5)  # filled value
    assert t.loc["2015-01-20", "in_harvest"] and not t.loc["2015-01-20", "in_cold_storage_release"]
    assert np.isnan(t.loc["2015-03-30", "y_ret_7"])                     # beyond the data


def test_msp_lead_and_phase_columns(refs):
    days = pd.date_range("2015-09-01", "2016-01-31")
    daily = make_daily({("Deesa", "Bajra"): pd.Series(1400.0, index=days),
                        ("Deesa", "Potato"): pd.Series(900.0, index=days),
                        ("Palanpur", "Bajra"): pd.Series(1300.0, index=days)})
    t = features.build_features(daily, make_weather(days), refs)
    bajra = t[t["commodity"] == "Bajra"].set_index("date")
    potato = t[t["commodity"] == "Potato"].set_index("date")

    assert bajra.loc["2015-09-30", "msp_rs_qtl"] == 1250 and bajra.loc["2015-10-01", "msp_rs_qtl"] == 1275
    assert bajra.loc["2015-10-01", "price_to_msp"] == pytest.approx(1400 / 1275)
    assert potato["price_to_msp"].isna().all()                          # no MSP for potato
    assert bajra.loc["2015-10-01", "x_palanpur_gap"] == pytest.approx(np.log(1400 / 1300))
    assert potato["x_palanpur_gap"].isna().all()                        # Palanpur has no potato series
    assert bajra["x_dhanera_gap"].isna().all()                          # no Dhanera data at all
    assert bajra.loc["2015-10-01", "in_harvest"] and not bajra["in_cold_storage_release"].any()
    assert bajra.loc["2015-11-01", "days_to_diwali"] == 10
    assert not (t["market"] == "Palanpur").any()                        # lead markets get no rows


def test_seasonal_change(refs):
    """seas_ret_h at t = last year's change from around t-365 to around t+h-365."""
    days = pd.date_range("2015-01-01", "2016-12-31")
    p = pd.Series(1000.0 * np.exp(0.3 * np.sin(2 * np.pi * days.dayofyear / 365)), index=days)
    t = features.build_features(make_daily({("Deesa", "Potato"): p}), make_weather(days), refs)
    t = t.set_index("date")
    day = pd.Timestamp("2016-03-01")
    row = t.loc[day]
    lp, d = np.log(p), pd.Timedelta(days=1)

    def week(center):                                                   # 7 days centred on a date
        return lp[center - 3 * d:center + 3 * d].mean()

    expected = week(day - 365 * d + 30 * d) - week(day - 365 * d)
    assert row["seas_ret_30"] == pytest.approx(expected, rel=1e-3)
    assert np.isnan(t.loc["2015-06-01", "seas_ret_7"])                 # no year of history yet


def test_splits():
    dates = pd.Series(pd.to_datetime(["2014-06-01", "2015-01-01", "2022-12-31", "2023-07-01",
                                      "2024-01-01", "2025-12-31", "2026-01-01"]))
    assert features.assign_split(dates).tolist() == ["warmup", "train", "train", "valid",
                                                     "test", "test", "live"]
    assert config.SPLITS["train"][1] < config.SPLITS["valid"][0] < config.SPLITS["test"][0]
