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


def test_xgb_walk_forward_learns_signal_and_beats_naive():
    t = synthetic_table()
    preds = models.evaluate(t, ["naive", "seasonal", "xgb"], [7], start="2023-01-01", end="2023-06-30")
    m = models.metrics(preds).xs(("valid", 7), level=("split", "horizon"))
    xgb, naive = m.loc[("xgb", "All")], m.loc[("naive", "All")]
    assert xgb["mape"] < naive["mape"] / 2
    assert xgb["da"] > 90 and np.isnan(naive["da"])
    assert 60 < xgb["coverage"] < 98                                  # 10-90 range roughly honest
    assert set(preds["model"]) == {"naive", "seasonal", "xgb"}
    assert preds.groupby("model").size().nunique() == 1               # same rows for every model


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
