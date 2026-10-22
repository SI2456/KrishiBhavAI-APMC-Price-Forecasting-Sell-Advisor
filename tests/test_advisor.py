"""Tests for the M5 sell/hold advisor and its back-test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krishibhav import advisor

STORAGE = {"Potato": 1.0, "Bajra": 0.0}
RATE = 0.0


def wide_row(price=1000.0, crop="Potato", split="test", date="2024-01-01", **by_h):
    """One decided-ready row. by_h: e.g. p50_7=..., p10_7=..., actual_7=..."""
    row = {"date": pd.Timestamp(date), "commodity": crop, "split": split, "price_now": price}
    for h in (7, 14, 30):
        row.update({f"p10_{h}": price * 0.95, f"p50_{h}": price, f"p90_{h}": price * 1.05,
                    f"actual_{h}": price})
    row.update(by_h)
    return row


def test_rules():
    rows = pd.DataFrame([
        # expected +65 at 30d minus 30 storage = +35 (> 3%), and p10 above today -> hold 30
        wide_row(p50_30=1065.0, p10_30=1010.0),
        # same gain but p10 below today -> sell half
        wide_row(p50_30=1065.0, p10_30=990.0),
        # +40 at 30d minus 30 storage = +10 (1%) -> below a 3% threshold -> sell now
        wide_row(p50_30=1040.0, p10_30=1010.0),
        # 7 days beats 30 days once costs are counted
        wide_row(p50_7=1045.0, p10_7=1001.0, p50_30=1065.0, p10_30=1010.0),
    ])
    out = advisor.decide(rows, threshold=0.03, storage=STORAGE, rate=RATE)
    assert out["action"].tolist() == ["hold", "sell_half", "sell_now", "hold"]
    assert out["best_h"].tolist() == [30, 30, 30, 7]
    assert out["expected_gain"].tolist() == pytest.approx([35.0, 35.0, 10.0, 38.0])


def test_interest_counts_as_cost():
    row = pd.DataFrame([wide_row(crop="Bajra", p50_30=1035.0, p10_30=1010.0)])   # bajra: no storage cost
    no_interest = advisor.decide(row, 0.03, STORAGE, rate=0.0)
    with_interest = advisor.decide(row, 0.03, STORAGE, rate=0.12)
    assert no_interest["action"].iloc[0] == "hold"
    assert with_interest["expected_gain"].iloc[0] == pytest.approx(35 - 1000 * 0.12 * 30 / 365)
    assert with_interest["action"].iloc[0] == "sell_now"              # 25.1 left, below 3% of 1000


def test_realised_revenue():
    rows = pd.DataFrame([
        wide_row(p50_30=1065.0, p10_30=1010.0, actual_30=1100.0),     # hold: 1100 - 30 storage
        wide_row(p50_30=1065.0, p10_30=990.0, actual_30=900.0),       # half: (1000 + 900 - 30) / 2
        wide_row(actual_30=2000.0),                                   # sell now: 1000
    ])
    out = advisor.realised(advisor.decide(rows, 0.03, STORAGE, RATE), STORAGE, RATE)
    assert out["revenue"].tolist() == pytest.approx([1070.0, 935.0, 1000.0])
    assert out["gain"].tolist() == pytest.approx([70.0, -65.0, 0.0])


def test_wide_forecasts_and_complete_rows():
    long = []
    for h in (7, 14, 30):
        long.append({"date": pd.Timestamp("2024-01-01"), "commodity": "Potato", "split": "test",
                     "reported": True, "horizon": h, "model": "xgb", "price_now": 1000.0,
                     "y_price": 1000.0 + h, "q10": -0.1, "q50": 0.0, "q90": 0.1,
                     "driver_1": "time of year", "effect_1": 1.5})
    long.append(dict(long[0], model="naive"))
    long.append(dict(long[0], date=pd.Timestamp("2024-01-02")))      # only 7d for this day
    wide = advisor.wide_forecasts(pd.DataFrame(long))
    assert len(wide) == 2
    row = wide.iloc[0]
    assert row["actual_14"] == 1014.0 and row["p50_30"] == pytest.approx(1000.0)
    assert row["p10_7"] == pytest.approx(1000 * np.exp(-0.1)) and row["driver_1_7"] == "time of year"
    assert len(advisor.complete_rows(wide)) == 1


def test_threshold_is_chosen_on_valid_only():
    rng = np.random.default_rng(0)
    rows = []
    for split in ("valid", "test"):
        for i in range(200):
            up = rng.normal(0, 40)
            rows.append(wide_row(split=split, crop="Bajra", date=f"2024-01-01",
                                 p50_30=1000 + up, p10_30=1000 + up - 20, actual_30=1000 + up + rng.normal(0, 30)))
    wide = pd.DataFrame(rows)
    best, sweep = advisor.choose_threshold(wide, storage=STORAGE, rate=RATE)
    shuffled = wide.copy()
    shuffled.loc[shuffled["split"] == "test", "actual_30"] = 0.0      # wreck the test outcomes
    assert advisor.choose_threshold(shuffled, storage=STORAGE, rate=RATE)[0] == best
    assert list(sweep["threshold_%"]) == [100 * t for t in advisor.THRESHOLDS]


def test_strategy_references():
    rows = pd.DataFrame([wide_row(p50_30=1065.0, p10_30=1010.0, actual_30=900.0, actual_7=1050.0)])
    out = advisor.strategy_gains(rows, 0.03, STORAGE, RATE)
    assert out["gain"].iloc[0] == pytest.approx(900 - 30 - 1000)       # advice was wrong here
    assert out["gain_hold_longest"].iloc[0] == pytest.approx(-130.0)
    assert out["gain_perfect"].iloc[0] == pytest.approx(43.0)          # best: sell at day 7
    assert (out["gain_perfect"] >= out["gain"]).all()


def test_advice_text_gujarati_and_lot():
    row = advisor.decide(pd.DataFrame([wide_row(
        p50_14=1080.0, p10_14=1010.0, p90_14=1150.0,
        driver_1_14="time of year", effect_1_14=-4.8, driver_2_14="arrivals", effect_2_14=2.0)]),
        0.03, STORAGE, RATE).iloc[0]
    t = advisor.advice_text(row, lot_qtl=50)
    assert t["action"] == "hold" and t["gu"]["advice"] == "14 દિવસ માલ રાખો"
    assert t["en"]["gain"] == "Expected extra after costs: Rs 3,300 on 50 quintal"   # (80 - 14) * 50
    assert "₹1,010 થી ₹1,150" in t["gu"]["forecast"]
    assert t["gu"]["reasons"][0] == "વર્ષનો સમય (મોસમ): ભાવ ઘટવા તરફ (-4.8%)"
    assert t["gu"]["reasons"][1].startswith("આવક: ભાવ વધવા તરફ")
    assert len(t["gu"]["reasons"]) == 2 and t["gu"]["note"]


def test_sell_now_shows_no_gain():
    row = advisor.decide(pd.DataFrame([wide_row()]), 0.03, STORAGE, RATE).iloc[0]
    t = advisor.advice_text(row, lot_qtl=10)
    assert t["gu"]["advice"] == "હમણાં વેચો" and "Rs 0 on 10 quintal" in t["en"]["gain"]


def test_bootstrap_ci():
    days = pd.date_range("2024-01-01", periods=120)
    flat = pd.DataFrame({"date": days, "gain": 5.0})
    mean, lo, hi = advisor.bootstrap_ci(flat)
    assert mean == lo == hi == 5.0
    noisy = pd.DataFrame({"date": days, "gain": np.random.default_rng(0).normal(0, 50, 120)})
    mean, lo, hi = advisor.bootstrap_ci(noisy)
    assert lo < mean < hi and lo < 0 < hi
