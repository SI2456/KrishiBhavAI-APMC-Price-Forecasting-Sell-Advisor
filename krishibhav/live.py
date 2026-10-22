"""Live advice: today's sell-or-hold advice for each Deesa crop.

  1. rebuild the feature table from the database (skip with rebuild=False)
  2. per horizon, train XGBoost on every row whose outcome is already known
     and save it to models/xgb_h<h>.json
  3. forecast each crop from its latest reported Deesa price: 10/50/90
     percentile price and top 3 drivers; the 10-90 range is shifted with the
     crop's latest CAL_MONTHS months of walk-forward errors (from `evaluate`),
     exactly as in the back-test
  4. apply the advisor rules and write data/processed/advice_latest.json for
     the web app, and print a Gujarati / English card per crop

A crop whose latest price is older than STALE_DAYS gets a warning; older than
MAX_AGE_DAYS it gets no advice at all ("no recent price").
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import advisor, config, features, models

log = logging.getLogger(__name__)
STALE_DAYS = 7
MAX_AGE_DAYS = 30
CROP_GU = {"Potato": "બટાકા", "Castor Seed": "એરંડા", "Groundnut": "મગફળી", "Bajra": "બાજરી"}
MARKET_GU = {"Deesa": "ડીસા"}


def today() -> date:
    return date.today()


def latest_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Last reported row per crop in the feature markets (the price the advice starts from)."""
    t = table[table["market"].isin(config.FEATURE_MARKETS)].sort_values("date")
    reported = t[t["reported"].astype(bool)]
    rows = reported.groupby(["market", "commodity"]).tail(1)
    return rows.sort_values(["market", "commodity"])


def live_offsets(preds: pd.DataFrame | None, crop: str, h: int, as_of: pd.Timestamp) -> dict:
    """Range shifts from the crop's most recent CAL_MONTHS months of walk-forward
    errors whose outcome was known by as_of."""
    none = {"lo": 0.0, "hi": 0.0, "rows": 0, "from": None, "to": None}
    if preds is None or preds.empty:
        return none
    p = preds[(preds["model"] == "xgb") & (preds["commodity"] == crop) & (preds["horizon"] == h)
              & preds["reported"].astype(bool) & preds["y_ret"].notna()
              & (preds["date"] + pd.Timedelta(days=h) <= as_of)]
    if p.empty:
        return none
    recent = p[p["date"] > p["date"].max() - pd.DateOffset(months=models.CAL_MONTHS)]
    lo, hi = models.range_offsets(recent)
    return {"lo": lo, "hi": hi, "rows": len(recent),
            "from": recent["date"].min().date().isoformat(), "to": recent["date"].max().date().isoformat()}


def forecast_latest(table: pd.DataFrame, preds: pd.DataFrame | None, horizons=None,
                    save_models: bool = True) -> tuple[pd.DataFrame, dict]:
    """One row per crop in the advisor's wide format (price_now, p10_h, p50_h,
    p90_h, driver_i_h, effect_i_h), plus training info per horizon."""
    horizons = horizons or config.HORIZONS
    as_of = table["date"].max()
    latest = latest_rows(table).copy()
    wide = latest[["date", "market", "commodity", "split"]].copy()
    wide["price_now"] = np.exp(latest["log_price"]).to_numpy()
    columns = models.feature_columns(table)
    info = {}
    for h in horizons:
        train = models.training_rows(table, as_of + pd.Timedelta(days=1), h)
        model = models.fit_xgb(train, h, columns)
        q = models.predict_xgb_model(model, latest, columns)
        drivers = models.top_drivers(models.shap_by_label(model, latest, columns))
        info[h] = {"trained_rows": len(train), "trained_until": train["date"].max().date().isoformat(),
                   "calibration": {}}
        if save_models:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.save_model(str(config.MODELS_DIR / f"xgb_h{h}.json"))
        for idx, crop in latest["commodity"].items():
            off = live_offsets(preds, crop, h, as_of)
            info[h]["calibration"][crop] = off
            q.loc[idx, "q10"] = min(q.loc[idx, "q10"] + off["lo"], q.loc[idx, "q50"])
            q.loc[idx, "q90"] = max(q.loc[idx, "q90"] + off["hi"], q.loc[idx, "q50"])
        for name in ("q10", "q50", "q90"):
            wide[f"p{name[1:]}_{h}"] = wide["price_now"] * np.exp(q[name])
        for col in drivers.columns:
            wide[f"{col}_{h}"] = drivers[col]
    return wide.reset_index(drop=True), info


def _r(x, digits=0):
    return None if x is None or pd.isna(x) else round(float(x), digits) if digits else int(round(float(x)))


def build_payload(decided: pd.DataFrame, info: dict, lot_qtl: float, as_of: pd.Timestamp) -> dict:
    crops = []
    for _, row in decided.iterrows():
        age = (pd.Timestamp(today()) - pd.Timestamp(row["date"])).days
        price_date = pd.Timestamp(row["date"]).date()
        stale_en = stale_gu = None
        if age > STALE_DAYS:
            stale_en = f"Latest Deesa price is from {price_date:%d %b %Y} ({age} days old)."
            stale_gu = f"ડીસાનો છેલ્લો ભાવ {price_date:%d-%m-%Y} નો છે ({age} દિવસ જૂનો)."
        entry = {
            "commodity": row["commodity"], "commodity_gu": CROP_GU.get(row["commodity"], row["commodity"]),
            "market": row["market"], "market_gu": MARKET_GU.get(row["market"], row["market"]),
            "price_date": price_date.isoformat(), "price_age_days": age, "stale": age > STALE_DAYS,
            "price_now": _r(row["price_now"]),
            "forecasts": {},
        }
        for h in advisor.horizons_in(decided):
            entry["forecasts"][str(h)] = {
                "p10": _r(row[f"p10_{h}"]), "p50": _r(row[f"p50_{h}"]), "p90": _r(row[f"p90_{h}"]),
                "expected_gain_rs_qtl": _r(row[f"gain_{h}"], 1),
                "drivers": [{"label": row[f"driver_{i}_{h}"],
                             "label_gu": advisor.GUJARATI.get(row[f"driver_{i}_{h}"], row[f"driver_{i}_{h}"]),
                             "effect_pct": _r(row[f"effect_{i}_{h}"], 1)}
                            for i in range(1, models.TOP_DRIVERS + 1) if isinstance(row.get(f"driver_{i}_{h}"), str)],
            }
        if age > MAX_AGE_DAYS:
            entry.update({
                "action": "no_recent_price", "best_h": None, "expected_gain_rs_qtl": None,
                "expected_gain_lot_rs": None,
                "text": {"en": {"advice": "No advice: no recent Deesa price", "note": stale_en},
                         "gu": {"advice": "સલાહ નથી: ડીસાનો તાજો ભાવ ઉપલબ્ધ નથી", "note": stale_gu}},
            })
        else:
            text = advisor.advice_text(row, lot_qtl)
            share = {"sell_now": 0.0, "sell_half": 0.5, "hold": 1.0}[row["action"]]
            if stale_en:
                text["en"]["stale"], text["gu"]["stale"] = stale_en, stale_gu
            entry.update({
                "action": row["action"], "best_h": int(row["best_h"]),
                "expected_gain_rs_qtl": _r(row["expected_gain"], 1),
                "expected_gain_lot_rs": _r(row["expected_gain"] * lot_qtl * share) if share else 0,
                "text": {"en": text["en"], "gu": text["gu"]},
            })
        crops.append(entry)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_as_of": pd.Timestamp(as_of).date().isoformat(),
        "lot_qtl": lot_qtl,
        "settings": {"storage_rs_qtl_day": config.STORAGE_COST_RS_QTL_DAY,
                     "interest_rate": config.INTEREST_RATE, "hold_threshold": config.HOLD_THRESHOLD},
        "models": {str(h): v for h, v in info.items()},
        "disclaimer": {"en": advisor.DISCLAIMER[0], "gu": advisor.DISCLAIMER[1]},
        "crops": crops,
    }


def card(entry: dict) -> str:
    gu, en = entry["text"]["gu"], entry["text"]["en"]
    lines = [f"== {entry['commodity_gu']} / {entry['commodity']}  ({entry['market']}, "
             f"price of {entry['price_date']}: Rs {entry['price_now']:,})",
             f"   {gu['advice']}  /  {en['advice']}"]
    for key in ("stale", "forecast", "gain"):
        if gu.get(key):
            lines += [f"   {gu[key]}", f"   {en[key]}"]
    for g, e in zip(gu.get("reasons", []), en.get("reasons", [])):
        lines.append(f"     - {g}  /  {e}")
    if entry["action"] == "no_recent_price" and gu.get("note"):
        lines += [f"   {gu['note']}", f"   {en['note']}"]
    return "\n".join(lines)


def advise(lot_qtl: float = 50.0, crop: str | None = None, rebuild: bool = True) -> dict:
    if rebuild:
        features.build()
    path = config.PROCESSED / "features.csv"
    if not path.exists():
        raise FileNotFoundError("No features.csv: run `python -m krishibhav build-features` first.")
    table = pd.read_csv(path, parse_dates=["date"])
    pred_path = config.PROCESSED / "predictions_m3.csv"
    preds = pd.read_csv(pred_path, parse_dates=["date"]) if pred_path.exists() else None
    if preds is None:
        log.warning("No predictions_m3.csv: ranges are not calibrated. Run `evaluate` once.")

    wide, info = forecast_latest(table, preds)
    if crop:
        wide = wide[wide["commodity"].str.lower() == crop.lower()]
    decided = advisor.decide(wide)
    payload = build_payload(decided, info, lot_qtl, table["date"].max())
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "advice_latest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Advice for a {lot_qtl:g} quintal lot (data up to {payload['data_as_of']}):\n")
    print("\n\n".join(card(e) for e in payload["crops"]))
    print(f"\n{advisor.DISCLAIMER[1]}\n{advisor.DISCLAIMER[0]}")
    log.info("Wrote %s", out)
    return payload
