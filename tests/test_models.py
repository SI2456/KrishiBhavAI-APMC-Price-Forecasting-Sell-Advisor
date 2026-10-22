"""Tests for M3: baselines, walk-forward XGBoost and the metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krishibhav import models


def synthetic_table(days=pd.date_range("2021-01-01", "2023-06-30"), seed=0):
    """Feature-table-shaped data where y_ret_7 = -0.8 * ret_7 + small noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for crop, level in [("Potato", 1000.0), ("Bajra", 2500.0)]:
        ret_7 = rng.normal(0, 0.05, len(days))
        rows.append(pd.DataFrame({
            "date": days, "market": "Deesa", "commodity": crop, "split": "train",
            "log_price": np.log(level), "ret_7": ret_7, "noise": rng.normal(size=len(days)),
            "seas_ret_7": 0.01, "reported": True,
            "y_ret_7": -0.8 * ret_7 + rng.normal(0, 0.005, len(days)),
        }))
    t = pd.concat(rows, ignore_index=True)
    t["y_price_7"] = np.exp(t["log_price"] + t["y_ret_7"])
    t.loc[t["date"] >= "2023-01-01", "split"] = "valid"
    return t


def test_training_rows_only_use_known_targets():
    t = synthetic_table()
    train = models.training_rows(t, pd.Timestamp("2023-03-01"), 7)
    assert train["date"].max() == pd.Timestamp("2023-02-21")       # 2023-02-21 + 7 < 2023-03-01
    assert (train["date"] + pd.Timedelta(days=7) < pd.Timestamp("2023-03-01")).all()


def test_baselines():
    t = synthetic_table()
    rows = t.head(3)
    assert (models.predict_naive(rows, 7)["q50"] == 0).all()
    assert (models.predict_seasonal(rows, 7)["q50"] == 0.01).all()
    rows = rows.assign(seas_ret_7=np.nan)
    assert (models.predict_seasonal(rows, 7)["q50"] == 0).all()     # no history -> naive


@pytest.fixture
def fast_xgb(monkeypatch):
    monkeypatch.setattr(models, "XGB_ROUNDS", 150)
    monkeypatch.setattr(models, "CAL_MONTHS", 3)


def test_xgb_walk_forward_learns_signal_and_beats_naive(fast_xgb):
    t = synthetic_table()
    importance = []
    preds = models.evaluate(t, ["naive", "seasonal", "xgb"], [7], start="2023-01-01", end="2023-06-30",
                            importance=importance)
    m = models.metrics(preds).xs(("valid", 7), level=("split", "horizon"))
    xgb, naive = m.loc[("xgb", "All")], m.loc[("naive", "All")]
    assert xgb["mape"] < naive["mape"] / 2
    assert xgb["da"] > 90 and np.isnan(naive["da"])
    assert 60 < xgb["coverage"] < 98                                  # 10-90 range roughly honest
    assert set(preds["model"]) == {"naive", "seasonal", "xgb"}
    assert preds.groupby("model").size().nunique() == 1               # same rows for every model
    assert preds["date"].min() == pd.Timestamp("2023-01-01")          # calibration months not scored
    xgb_rows = preds[preds["model"] == "xgb"]
    assert (xgb_rows["driver_1"] == "price change over the last two weeks").mean() > 0.9
    # the top driver is the signal itself, so it points the same way as the forecast
    assert (np.sign(xgb_rows["effect_1"]) == np.sign(xgb_rows["q50"])).mean() > 0.8
    imp = models.importance_table(importance)
    assert imp.loc[(7, "Potato")].idxmax() == "price change over the last two weeks"


def test_metrics_by_hand():
    preds = pd.DataFrame({
        "split": "test", "horizon": 7, "model": "xgb", "commodity": "Potato", "reported": True,
        "y_price": [100.0, 200.0, 100.0, 100.0], "pred_price": [110.0, 180.0, 100.0, 50.0],
        "y_ret": [0.1, -0.1, 0.0, 0.2], "q50": [0.05, 0.02, 0.01, -0.1],
        "q10": [0.0, -0.2, -0.1, -0.3], "q90": [0.2, 0.0, 0.1, 0.0],
    })
    preds = pd.concat([preds, preds.assign(reported=False, y_price=1.0)])  # unreported rows ignored
    m = models.metrics(preds).loc[("test", 7, "xgb", "Potato")]
    assert m["n"] == 4
    assert m["mape"] == pytest.approx(100 * (0.1 + 0.1 + 0 + 0.5) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400 + 0 + 2500) / 4), abs=0.01)
    assert m["da"] == pytest.approx(100 / 3, abs=0.01)                # zero move left out; 1 of 3 right
    assert m["coverage"] == 75.0                                      # last actual is above q90


def test_results_table_bolds_best():
    m = pd.DataFrame({
        "split": "test", "horizon": [7, 7], "model": ["naive", "xgb"], "commodity": "All",
        "n": 10, "mape": [5.0, 4.0], "rmse": 1.0, "da": [np.nan, 61.0], "coverage": np.nan,
    }).set_index(["split", "horizon", "model", "commodity"])
    text = models.results_table(m, "test")
    assert "| naive | 5.0% | – |" in text and "| xgb | **4.0%** | **61.0%** |" in text


def test_driver_labels():
    assert models.driver_label("doy_sin") == models.driver_label("week") == "time of year"
    assert models.driver_label("ret_365") == "price vs a year ago"
    assert models.driver_label("ret_7") == "price change over the last two weeks"
    assert models.driver_label("price_to_msp") == "price vs MSP"
    assert models.driver_label("reported") == "days since the last report"
    assert models.driver_label("in_harvest") == "crop season (sowing / harvest)"
    assert models.driver_label("something_new") == "something_new"


def calibration_frame(seed=0, months=24, width=0.01):
    """Too-narrow raw ranges (+-1%) around 0 for returns that vary by ~5%."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-01", periods=months * 30, freq="D")
    return pd.DataFrame({"date": days, "commodity": "Potato", "reported": True,
                         "y_ret": rng.normal(0, 0.05, len(days)),
                         "q10": -width, "q50": 0.0, "q90": width})


def test_calibration_reaches_about_80_percent():
    out = models.calibrate(calibration_frame(), h=7, months=12)
    first = out["date"] < "2022-02-01"
    assert (out.loc[first, "q10"] == -0.01).all()                      # no history yet: unchanged
    later = out[out["date"] >= "2023-01-01"]
    inside = (later["y_ret"] >= later["q10"]) & (later["y_ret"] <= later["q90"])
    assert 0.72 < inside.mean() < 0.88
    raw_inside = (later["y_ret"] >= later["q10_raw"]) & (later["y_ret"] <= later["q90_raw"])
    assert raw_inside.mean() < 0.3
    assert (out["q10"] <= out["q50"]).all() and (out["q90"] >= out["q50"]).all()


def test_calibration_uses_only_known_outcomes():
    base = calibration_frame()
    changed = base.copy()
    month = pd.Timestamp("2022-10-01")
    changed.loc[changed["date"] >= month - pd.Timedelta(days=7), "y_ret"] *= 10   # not yet known
    a = models.calibrate(base, h=7, months=12)
    b = models.calibrate(changed, h=7, months=12)
    in_month = (a["date"] >= month) & (a["date"] < month + pd.DateOffset(months=1))
    pd.testing.assert_series_equal(a.loc[in_month, "q90"], b.loc[in_month, "q90"])


def test_diebold_mariano():
    rng = np.random.default_rng(1)
    better = rng.normal(-0.01, 0.01, 400)                              # A clearly lower loss
    stat, p = models.diebold_mariano(better, h=7)
    assert stat < -3 and p < 0.01
    stat, p = models.diebold_mariano(rng.normal(0, 0.01, 400), h=7)
    assert p > 0.05
    assert np.isnan(models.diebold_mariano(np.array([0.1, 0.2]), h=7)[0])   # too short


def test_dm_table_all_crops():
    days = pd.date_range("2024-01-01", periods=60)
    rows = []
    for crop in ["Potato", "Bajra"]:
        for model, err in [("xgb", 0.05), ("naive", 0.10)]:
            rows.append(pd.DataFrame({
                "date": days, "commodity": crop, "split": "test", "horizon": 7, "model": model,
                "reported": True, "y_price": 100.0,
                "pred_price": 100.0 * (1 + err + np.random.default_rng(0).normal(0, 0.01, 60))}))
    dm = models.dm_table(pd.concat(rows, ignore_index=True)).set_index("commodity")
    assert set(dm.index) == {"Potato", "Bajra", "All"}
    assert dm.loc["All", "mape_diff_pts"] == pytest.approx(-5.0, abs=0.5)
    assert dm.loc["All", "p_value"] < 0.01 and dm.loc["All", "n"] == 60
