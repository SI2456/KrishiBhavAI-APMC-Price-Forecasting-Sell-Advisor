"""M5: sell-or-hold advice and its revenue back-test.

For each horizon h the expected net gain of holding one quintal is

    G_h = P50(t+h) - P(t) - c_s * h - P(t) * r * h / 365

P(t) today's price, P50 the median forecast, c_s storage cost per quintal per
day, r the yearly interest rate on money owed. With the best horizon h*:

    best G_h > threshold * P(t) and P10(t+h*) > P(t)  -> hold for h* days
    best G_h > threshold * P(t) but P10(t+h*) <= P(t)  -> sell half now, hold half
    otherwise                                           -> sell now

Back-test: replay every reported day in the walk-forward predictions (2023 to
2025, never trained on), follow the advice for one quintal, and compare the
money received (after storage and interest) with selling on the day of arrival.
The threshold is chosen on 2023 (valid) only, then applied to 2024-25 (test).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
ACTIONS = ["sell_now", "sell_half", "hold"]
THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]

TEXT = {
    "sell_now": ("Sell now", "હમણાં વેચો"),
    "sell_half": ("Sell half now, hold half for {h} days", "અડધો માલ હમણાં વેચો, અડધો {h} દિવસ રાખો"),
    "hold": ("Hold for {h} days", "{h} દિવસ માલ રાખો"),
}
# Plain-label drivers (models.DRIVER_LABELS) in Gujarati. CHECK WORDING WITH USERS.
GUJARATI = {
    "price level": "હાલનું ભાવ સ્તર",
    "price vs a year ago": "ગયા વર્ષની સરખામણીએ ભાવ",
    "price change over the last month": "છેલ્લા મહિનામાં ભાવનો ફેરફાર",
    "price change over the last two weeks": "છેલ્લા બે અઠવાડિયામાં ભાવનો ફેરફાર",
    "price vs its recent average": "તાજેતરની સરેરાશ સામે ભાવ",
    "price volatility": "ભાવની વધઘટ",
    "gap between max and min price": "ઊંચા અને નીચા ભાવ વચ્ચેનો તફાવત",
    "days since the last report": "છેલ્લા ભાવ પછીના દિવસો",
    "arrivals": "આવક",
    "rainfall": "વરસાદ",
    "temperature": "તાપમાન",
    "crop water demand": "પાકની પાણીની જરૂરિયાત",
    "price vs MSP": "ટેકાના ભાવ (MSP) સામે ભાવ",
    "day of week": "અઠવાડિયાનો દિવસ",
    "time of year": "વર્ષનો સમય (મોસમ)",
    "festival season": "તહેવારની મોસમ",
    "yard holiday": "યાર્ડની રજા",
    "policy event": "સરકારી નીતિમાં ફેરફાર",
    "crop season (sowing / harvest)": "વાવણી / કાપણીની મોસમ",
    "Deesa vs Palanpur price": "ડીસા અને પાલનપુરના ભાવનો તફાવત",
    "Deesa vs Dhanera price": "ડીસા અને ધાનેરાના ભાવનો તફાવત",
    "last year's pattern for these weeks": "ગયા વર્ષે આ અઠવાડિયાઓમાં ભાવનો રુખ",
    "crop": "પાક",
}
DISCLAIMER = ("This is an estimate from past prices, not a guarantee.",
              "આ ફક્ત જૂના ભાવ પરથી અંદાજ છે, ભાવની ખાતરી નથી.")


# ---------------------------------------------------------------- inputs
def wide_forecasts(preds: pd.DataFrame, model: str = "xgb") -> pd.DataFrame:
    """predictions_m3 (long) -> one row per date and crop with price_now and,
    per horizon h: p10_h, p50_h, p90_h (prices) and actual_h (realised price)."""
    p = preds[(preds["model"] == model) & preds["reported"].astype(bool)].copy()
    for q in ("q10", "q50", "q90"):
        p[f"p{q[1:]}"] = p["price_now"] * np.exp(p[q])
    p = p.rename(columns={"y_price": "actual"})
    keep = ["p10", "p50", "p90", "actual"] + [c for c in p.columns if c.startswith(("driver_", "effect_"))]
    wide = p.pivot_table(index=["date", "commodity", "split", "price_now"], columns="horizon",
                         values=keep, aggfunc="first")
    wide.columns = [f"{name}_{h}" for name, h in wide.columns]
    return wide.reset_index().sort_values(["commodity", "date"]).reset_index(drop=True)


def horizons_in(wide: pd.DataFrame) -> list[int]:
    return sorted(int(c.split("_")[1]) for c in wide.columns if c.startswith("p50_"))


# ---------------------------------------------------------------- rules
def holding_cost(price_now, crop, h, storage=None, rate=None):
    storage = config.STORAGE_COST_RS_QTL_DAY if storage is None else storage
    rate = config.INTEREST_RATE if rate is None else rate
    cs = pd.Series(crop).map(storage).fillna(0.0).to_numpy() if not np.isscalar(crop) \
        else storage.get(crop, 0.0)
    return cs * h + price_now * rate * h / 365


def decide(wide: pd.DataFrame, threshold: float | None = None, storage=None, rate=None) -> pd.DataFrame:
    """Add gain_<h> (expected net gain per quintal), best_h, action, expected_gain."""
    threshold = config.HOLD_THRESHOLD if threshold is None else threshold
    out = wide.copy()
    hs = horizons_in(out)
    for h in hs:
        out[f"gain_{h}"] = out[f"p50_{h}"] - out["price_now"] - holding_cost(
            out["price_now"].to_numpy(), out["commodity"], h, storage, rate)
    gains = out[[f"gain_{h}" for h in hs]].to_numpy()
    best = np.nanargmax(np.where(np.isnan(gains), -np.inf, gains), axis=1)
    out["best_h"] = np.asarray(hs)[best]
    out["expected_gain"] = gains[np.arange(len(out)), best]
    p10_best = out[[f"p10_{h}" for h in hs]].to_numpy()[np.arange(len(out)), best]
    worth_it = out["expected_gain"] > threshold * out["price_now"]
    safe = p10_best > out["price_now"]
    out["action"] = np.where(worth_it & safe, "hold", np.where(worth_it, "sell_half", "sell_now"))
    return out


def realised(decided: pd.DataFrame, storage=None, rate=None) -> pd.DataFrame:
    """Money per quintal from following the advice, and the gain over selling on
    arrival (which earns price_now)."""
    out = decided.copy()
    h = out["best_h"].to_numpy()
    actual = np.choose(np.searchsorted(horizons_in(out), h),
                       [out[f"actual_{x}"].to_numpy() for x in horizons_in(out)])
    held = actual - holding_cost(out["price_now"].to_numpy(), out["commodity"], h, storage, rate)
    share_held = out["action"].map({"sell_now": 0.0, "sell_half": 0.5, "hold": 1.0}).to_numpy()
    out["revenue"] = (1 - share_held) * out["price_now"] + share_held * held
    out["gain"] = out["revenue"] - out["price_now"]
    return out


# ---------------------------------------------------------------- back-test
def complete_rows(wide: pd.DataFrame) -> pd.DataFrame:
    """Days where every horizon has a forecast and a realised price (a fair,
    identical sample for every strategy)."""
    cols = [f"{k}_{h}" for h in horizons_in(wide) for k in ("p10", "p50", "actual")]
    return wide.dropna(subset=cols)


def strategy_gains(wide: pd.DataFrame, threshold: float, storage=None, rate=None) -> pd.DataFrame:
    """Gain per quintal for the advisor and two reference strategies."""
    adv = realised(decide(wide, threshold, storage, rate), storage, rate)
    hs = horizons_in(wide)
    longest = max(hs)
    hold_all = wide[f"actual_{longest}"] - holding_cost(
        wide["price_now"].to_numpy(), wide["commodity"], longest, storage, rate) - wide["price_now"]
    perfect = np.max(np.column_stack(
        [np.zeros(len(wide))] +
        [(wide[f"actual_{h}"] - holding_cost(wide["price_now"].to_numpy(), wide["commodity"], h, storage, rate)
          - wide["price_now"]).to_numpy() for h in hs]), axis=1)
    adv["gain_hold_longest"] = hold_all.to_numpy()
    adv["gain_perfect"] = perfect
    return adv


def summarise(adv: pd.DataFrame, by=("split", "commodity")) -> pd.DataFrame:
    def one(g):
        held = g[g["action"] != "sell_now"]
        return pd.Series({
            "days": len(g),
            "hold_%": 100 * (g["action"] == "hold").mean(),
            "sell_half_%": 100 * (g["action"] == "sell_half").mean(),
            "sell_now_%": 100 * (g["action"] == "sell_now").mean(),
            "avg_gain_rs_qtl": g["gain"].mean(),
            "avg_gain_%": 100 * (g["gain"] / g["price_now"]).mean(),
            "gain_when_holding_rs": held["gain"].mean() if len(held) else np.nan,
            "holds_that_lost_%": 100 * (held["gain"] < 0).mean() if len(held) else np.nan,
            "worst_5%_rs": g["gain"].quantile(0.05),
            "always_hold_30d_rs": g["gain_hold_longest"].mean(),
            "perfect_foresight_rs": g["gain_perfect"].mean(),
        })
    parts = [adv.groupby(list(by)).apply(one, include_groups=False)]
    overall = adv.groupby(list(by[:-1])).apply(one, include_groups=False) if len(by) > 1 else None
    if overall is not None:
        overall[by[-1]] = "All"
        parts.append(overall.set_index(by[-1], append=True))
    return pd.concat(parts).sort_index().round(1)


def bootstrap_ci(adv: pd.DataFrame, block: int = 30, draws: int = 5000, seed: int = 0
                 ) -> tuple[float, float, float]:
    """Mean daily gain (averaged across crops) and a 95% moving-block bootstrap
    interval; blocks of `block` days keep the day-to-day correlation."""
    d = adv.groupby("date")["gain"].mean().sort_index().to_numpy()
    n = len(d)
    if n < 2 * block:
        return float(d.mean()) if n else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    means = [np.concatenate([d[i:i + block] for i in rng.choice(starts, n // block + 1)])[:n].mean()
             for _ in range(draws)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def choose_threshold(wide: pd.DataFrame, split: str = "valid", storage=None, rate=None
                     ) -> tuple[float, pd.DataFrame]:
    """Pick the threshold with the highest average gain on `split` only."""
    rows = []
    sample = wide[wide["split"] == split]
    for th in THRESHOLDS:
        adv = realised(decide(sample, th, storage, rate), storage, rate)
        rows.append({"threshold_%": 100 * th, "avg_gain_rs_qtl": adv["gain"].mean(),
                     "hold_or_half_%": 100 * (adv["action"] != "sell_now").mean()})
    sweep = pd.DataFrame(rows).round(2)
    best = sweep.loc[sweep["avg_gain_rs_qtl"].idxmax(), "threshold_%"] / 100
    return float(best), sweep


# ---------------------------------------------------------------- advice text
def advice_text(row: pd.Series, lot_qtl: float = 1.0) -> dict:
    """English and Gujarati advice for one decided row (output of decide())."""
    h = int(row["best_h"])
    action = row["action"]
    en_action, gu_action = (t.format(h=h) for t in TEXT[action])
    p10, p50, p90 = (row[f"p{q}_{h}"] for q in (10, 50, 90))
    share_held = {"sell_now": 0.0, "sell_half": 0.5, "hold": 1.0}[action]
    gain_lot = row["expected_gain"] * lot_qtl * share_held if share_held else 0.0
    reasons_en, reasons_gu = [], []
    for i in (1, 2, 3):
        label, effect = row.get(f"driver_{i}_{h}"), row.get(f"effect_{i}_{h}")
        if not isinstance(label, str) or pd.isna(effect):
            continue
        up = effect >= 0
        reasons_en.append(f"{label}: pushes the price {'up' if up else 'down'} ({effect:+.1f}%)")
        reasons_gu.append(f"{GUJARATI.get(label, label)}: ભાવ {'વધવા' if up else 'ઘટવા'} તરફ ({effect:+.1f}%)")
    return {
        "action": action,
        "en": {
            "advice": en_action,
            "forecast": f"Price in {h} days: about Rs {p50:,.0f} (likely Rs {p10:,.0f} to Rs {p90:,.0f}); "
                        f"today Rs {row['price_now']:,.0f}",
            "gain": f"Expected extra after costs: Rs {gain_lot:,.0f} on {lot_qtl:g} quintal",
            "reasons": reasons_en, "note": DISCLAIMER[0],
        },
        "gu": {
            "advice": gu_action,
            "forecast": f"{h} દિવસ પછી અંદાજિત ભાવ ₹{p50:,.0f} (₹{p10:,.0f} થી ₹{p90:,.0f}); આજનો ભાવ ₹{row['price_now']:,.0f}",
            "gain": f"ખર્ચ બાદ અંદાજિત વધારાનો નફો: ₹{gain_lot:,.0f} ({lot_qtl:g} ક્વિન્ટલ પર)",
            "reasons": reasons_gu, "note": DISCLAIMER[1],
        },
    }


def card_markdown(row: pd.Series, lot_qtl: float) -> str:
    t = advice_text(row, lot_qtl)
    lines = [f"**{row['commodity']}, {pd.Timestamp(row['date']):%d %b %Y}**", "",
             f"- **{t['gu']['advice']}** / {t['en']['advice']}",
             f"- {t['gu']['forecast']}", f"- {t['en']['forecast']}",
             f"- {t['gu']['gain']}", f"- {t['en']['gain']}"]
    lines += [f"  - {g} / {e}" for g, e in zip(t["gu"]["reasons"], t["en"]["reasons"])]
    lines.append(f"- _{t['gu']['note']}_")
    return "\n".join(lines)


# ---------------------------------------------------------------- command
def md(df: pd.DataFrame, index: bool = True) -> str:
    """Small markdown table writer (no extra package needed)."""
    d = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def run(lot_qtl: float = 50.0) -> pd.DataFrame:
    path = config.PROCESSED / "predictions_m3.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m krishibhav evaluate` first (it writes predictions_m3.csv).")
    preds = pd.read_csv(path, parse_dates=["date"])
    wide = complete_rows(wide_forecasts(preds))
    threshold, sweep = choose_threshold(wide)
    adv = strategy_gains(wide, threshold)
    table = summarise(adv)
    adv.to_csv(config.PROCESSED / "advisor_backtest.csv", index=False)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS / "m5_backtest.csv")

    ci_lines = []
    for split in ("valid", "test"):
        mean, lo, hi = bootstrap_ci(adv[adv["split"] == split])
        verdict = "above zero" if lo > 0 else "includes zero (not significant at 5%)"
        ci_lines.append(f"| {split} | {mean:.1f} | {lo:.1f} to {hi:.1f} | {verdict} |")
    by_year = summarise(adv.assign(year=adv["date"].dt.year), by=("year", "commodity"))
    cols = ["days", "hold_%", "sell_half_%", "sell_now_%", "avg_gain_rs_qtl", "avg_gain_%",
            "holds_that_lost_%", "worst_5%_rs", "always_hold_30d_rs", "perfect_foresight_rs"]
    examples = []
    test = adv[adv["split"] == "test"]
    for crop in sorted(test["commodity"].unique()):
        held = test[(test["commodity"] == crop) & (test["action"] != "sell_now")]
        pick = held.iloc[len(held) // 2] if len(held) else test[test["commodity"] == crop].iloc[-1]
        examples.append(card_markdown(pick, lot_qtl))

    text = f"""# M5: sell-or-hold advisor back-test

Every reported day in 2023-2025 (walk-forward forecasts the model never trained
on): follow the advice for 1 quintal and compare with selling on arrival.
Gains are after storage cost ({config.STORAGE_COST_RS_QTL_DAY} Rs/qtl/day) and
{100 * config.INTEREST_RATE:.0f}% yearly interest. Only days where every horizon has a
forecast and a realised price are used, so all strategies see the same days.

## Threshold, chosen on 2023 only

{md(sweep, index=False)}

Chosen: **{100 * threshold:g}%** of today's price (the plan's starting value was 3%).

## Results (Rs per quintal vs selling on arrival)

{md(table[cols])}

Is the gain real? Mean daily gain (Rs/qtl, averaged across crops) with a 95%
moving-block bootstrap interval (30-day blocks):

| split | mean gain | 95% interval | |
|---|---|---|---|
""" + "\n".join(ci_lines) + f"""

`always_hold_30d_rs`: always keep the crop 30 days. `perfect_foresight_rs`: the
best possible choice each day if the future were known (an upper bound).

## By year

{md(by_year[cols])}

## Example advice (test period, {lot_qtl:g} quintal lot)

""" + "\n\n".join(examples) + "\n"
    (config.REPORTS / "m5_backtest.md").write_text(text, encoding="utf-8")
    print(text)
    return table
