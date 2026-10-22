"""Tests for the live `advise` command."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from krishibhav import advisor, config, live, models
from tests.test_models import synthetic_table


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "HORIZONS", [7])
    monkeypatch.setattr(models, "XGB_ROUNDS", 60)
    monkeypatch.setattr(live, "today", lambda: date(2023, 7, 1))
    config.PROCESSED.mkdir()

    t = synthetic_table()
    t = t[~((t["commodity"] == "Bajra") & (t["date"] > "2023-04-30"))]   # Bajra: no price for 2 months
    t.loc[(t["commodity"] == "Potato") & (t["date"] == "2023-06-30"), "reported"] = False
    t.to_csv(config.PROCESSED / "features.csv", index=False)

    rng = np.random.default_rng(0)
    days = pd.date_range("2022-01-01", "2023-06-30")
    preds = pd.DataFrame({"date": days, "commodity": "Potato", "horizon": 7, "model": "xgb",
                          "reported": True, "y_ret": rng.normal(0, 0.05, len(days)),
                          "q10": -0.01, "q50": 0.0, "q90": 0.01})
    preds.to_csv(config.PROCESSED / "predictions_m3.csv", index=False)
    return t, preds


def test_latest_rows(setup):
    t, _ = setup
    rows = live.latest_rows(t.assign(market=np.where(t.index % 50 == 0, "Palanpur", t["market"])))
    latest = rows.set_index("commodity")["date"]
    assert latest["Potato"] == pd.Timestamp("2023-06-29")                  # 30 Jun was not reported
    assert latest["Bajra"] == pd.Timestamp("2023-04-30")
    assert set(rows["market"]) == {"Deesa"}


def test_live_offsets_use_recent_known_errors(setup):
    _, preds = setup
    as_of = pd.Timestamp("2023-06-30")
    off = live.live_offsets(preds, "Potato", 7, as_of)
    assert off["to"] == "2023-06-23"                                       # 23 Jun + 7 days = as_of
    assert off["from"] > "2022-06-23" and off["rows"] > 300
    assert off["lo"] < -0.03 and off["hi"] > 0.03                          # +-1% raw range was far too narrow
    assert live.live_offsets(preds, "Bajra", 7, as_of)["rows"] == 0       # no errors for Bajra: no shift
    assert live.live_offsets(None, "Potato", 7, as_of)["lo"] == 0.0


def test_advise_end_to_end(setup):
    payload = live.advise(lot_qtl=20, rebuild=False)
    saved = json.loads((config.PROCESSED / "advice_latest.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(payload, ensure_ascii=False))
    assert saved["data_as_of"] == "2023-06-30" and saved["lot_qtl"] == 20
    assert (config.MODELS_DIR / "xgb_h7.json").exists()

    crops = {c["commodity"]: c for c in saved["crops"]}
    potato, bajra = crops["Potato"], crops["Bajra"]
    assert potato["action"] in advisor.ACTIONS and potato["best_h"] == 7
    assert potato["price_date"] == "2023-06-29" and not potato["stale"]
    f = potato["forecasts"]["7"]
    assert f["p10"] <= f["p50"] <= f["p90"] and len(f["drivers"]) == 3
    assert f["drivers"][0]["label_gu"] == advisor.GUJARATI.get(f["drivers"][0]["label"], f["drivers"][0]["label"])
    assert potato["text"]["gu"]["advice"] and potato["commodity_gu"] == "બટાકા"
    assert saved["models"]["7"]["calibration"]["Potato"]["rows"] > 300

    assert bajra["action"] == "no_recent_price" and bajra["price_age_days"] == 62
    assert bajra["text"]["gu"]["advice"].startswith("સલાહ નથી")


def test_advise_one_crop(setup):
    payload = live.advise(lot_qtl=5, crop="potato", rebuild=False)
    assert [c["commodity"] for c in payload["crops"]] == ["Potato"]
