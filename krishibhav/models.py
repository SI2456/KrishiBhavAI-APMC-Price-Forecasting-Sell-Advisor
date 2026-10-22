"""M3: baselines and XGBoost, evaluated walk-forward on the feature table.

Models (each predicts the log return to t+h; price = P[t] * exp(return)):
  naive     P[t+h] = P[t]                                (return 0)
  seasonal  P[t+h] = P[t] * last year's change over the same window (seas_ret_h)
  xgb       one XGBoost model for all crops, quantile loss -> 10th / 50th / 90th
            percentile; the median is the point forecast

Walk-forward: for every month from the start of `valid` to the end of `test`,
XGBoost is retrained on rows whose target was already known before that month
began (date + h < month start) and predicts the month. Nothing is tuned on
`test`. Errors are measured on days with a real report at t.

Explanations: each xgb forecast carries its top 3 drivers (SHAP values of the
median forecast, related features summed into one plain label, e.g. sin/cos
of the day of year -> "time of year"), as label and effect in % price change.

Calibrated ranges: the raw 10-90 range is shifted per crop and horizon so that
about 10% of outcomes fall below it and 10% above, using only the previous
CAL_MONTHS months of out-of-sample errors whose outcome was already known.
The walk-forward starts CAL_MONTHS early to have that history; those extra
months are not scored. q10/q90 are calibrated, q10_raw/q90_raw are not.

Metrics per model, horizon, split and crop:
  MAPE %       mean |predicted - actual| / actual price
  RMSE         Rs/quintal
  DA %         directional accuracy: did it call up vs down (days with no
               change, and the naive model, are left out)
  coverage %   share of actual returns inside the calibrated 10-90 range (~80 is
               honest); coverage_raw % before calibration
Diebold-Mariano: is xgb's absolute % error significantly lower than naive's?
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, features

log = logging.getLogger(__name__)
MODELS = ["naive", "seasonal", "xgb"]
QUANTILES = [0.1, 0.5, 0.9]
XGB_PARAMS = {
    "objective": "reg:quantileerror", "quantile_alpha": np.array(QUANTILES),
    "tree_method": "hist", "max_depth": 5, "eta": 0.03, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 10, "seed": 0,
}
XGB_ROUNDS = 500
CAL_MONTHS = 12          # months of past out-of-sample errors used to calibrate ranges
CAL_MIN_ROWS = 30        # fewer than this for a crop -> range left as the model gave it
TOP_DRIVERS = 3

# Feature -> plain label for explanations. Features sharing a label are summed.
DRIVER_LABELS = [
    ("log_price", "price level"),
    ("ret_365", "price vs a year ago"),
    ("ret_30", "price change over the last month"),
    ("ret_", "price change over the last two weeks"),
    ("ma_gap_", "price vs its recent average"),
    ("vol_", "price volatility"),
    ("spread", "gap between max and min price"),
    ("reported", "days since the last report"),
    ("days_since_report", "days since the last report"),
    ("arr", "arrivals"),
    ("rain_", "rainfall"),
    ("tmean_", "temperature"),
    ("tmax_", "temperature"),
    ("et0_", "crop water demand"),
    ("msp_rs_qtl", "price vs MSP"),
    ("price_to_msp", "price vs MSP"),
    ("dow", "day of week"),
    ("month", "time of year"),
    ("week", "time of year"),
    ("doy_", "time of year"),
    ("festival", "festival season"),
    ("days_to_diwali", "festival season"),
    ("apmc_holiday", "yard holiday"),
    ("policy", "policy event"),
    ("in_", "crop season (sowing / harvest)"),
    ("x_palanpur_", "Deesa vs Palanpur price"),
    ("x_dhanera_", "Deesa vs Dhanera price"),
    ("seas_ret_", "last year's pattern for these weeks"),
    ("crop", "crop"),
]


def driver_label(feature: str) -> str:
    for prefix, label in DRIVER_LABELS:
        if feature == prefix or feature.startswith(prefix):
            return label
    return feature


# ---------------------------------------------------------------- data
def feature_columns(table: pd.DataFrame) -> list[str]:
    skip = set(features.ID_COLS)
    return [c for c in table.columns if c not in skip and not c.startswith("y_")]


def to_matrix(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = rows[columns].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X["crop"] = pd.Categorical(rows["commodity"], categories=sorted(config.COMMODITIES))
    return X


def training_rows(table: pd.DataFrame, month_start: pd.Timestamp, h: int) -> pd.DataFrame:
    """Rows whose target (price at t+h) was known before month_start."""
    cutoff = month_start - pd.Timedelta(days=h + 1)
    return table[(table["date"] <= cutoff) & table[f"y_ret_{h}"].notna()]


# ---------------------------------------------------------------- models
def predict_naive(rows: pd.DataFrame, h: int) -> pd.DataFrame:
    return pd.DataFrame({"q10": np.nan, "q50": 0.0, "q90": np.nan}, index=rows.index)


def predict_seasonal(rows: pd.DataFrame, h: int) -> pd.DataFrame:
    # Without a year of history it falls back to the naive forecast.
    return pd.DataFrame({"q10": np.nan, "q50": rows[f"seas_ret_{h}"].fillna(0.0), "q90": np.nan},
                        index=rows.index)


def fit_xgb(train: pd.DataFrame, h: int, columns: list[str]):
    import xgboost as xgb
    d = xgb.DMatrix(to_matrix(train, columns), train[f"y_ret_{h}"], enable_categorical=True)
    return xgb.train(XGB_PARAMS, d, XGB_ROUNDS)


def predict_xgb_model(model, rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    import xgboost as xgb
    q = model.predict(xgb.DMatrix(to_matrix(rows, columns), enable_categorical=True))
    q = np.sort(np.asarray(q).reshape(len(rows), -1), axis=1)       # quantiles must not cross
    return pd.DataFrame(q, columns=["q10", "q50", "q90"], index=rows.index)


def shap_by_label(model, rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """SHAP values of the median forecast, summed per plain label (rows x labels)."""
    import xgboost as xgb
    X = to_matrix(rows, columns)
    contribs = model.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    median = contribs[:, QUANTILES.index(0.5), :-1] if contribs.ndim == 3 else contribs[:, :-1]
    per_feature = pd.DataFrame(median, columns=list(X.columns), index=rows.index)
    return per_feature.T.groupby([driver_label(c) for c in X.columns]).sum().T


def top_drivers(shap: pd.DataFrame, k: int = TOP_DRIVERS) -> pd.DataFrame:
    """For each row, the k labels with the largest |SHAP|: driver_i and effect_i
    (effect in % price change, + pushes the forecast up)."""
    values = shap.to_numpy()
    order = np.argsort(-np.abs(values), axis=1)[:, :k]
    out = {}
    for i in range(k):
        idx = order[:, i]
        out[f"driver_{i + 1}"] = np.asarray(shap.columns)[idx]
        out[f"effect_{i + 1}"] = 100 * (np.exp(values[np.arange(len(values)), idx]) - 1)
    return pd.DataFrame(out, index=shap.index)


def walk_forward_xgb(table: pd.DataFrame, rows: pd.DataFrame, h: int,
                     importance: list | None = None) -> pd.DataFrame:
    columns = feature_columns(table)
    months = sorted(rows["date"].dt.to_period("M").unique())
    parts = []
    for month in months:
        start = month.to_timestamp()
        target = rows[rows["date"].dt.to_period("M") == month]
        train = training_rows(table, start, h)
        model = fit_xgb(train, h, columns)
        shap = shap_by_label(model, target, columns)
        parts.append(predict_xgb_model(model, target, columns).join(top_drivers(shap)))
        if importance is not None:
            importance.append(shap.abs().assign(commodity=target["commodity"].to_numpy(),
                                                date=target["date"].to_numpy(), horizon=h))
        log.info("xgb h=%d %s: trained on %d rows up to %s, predicted %d",
                 h, month, len(train), train["date"].max().date(), len(target))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["q10", "q50", "q90"])


PREDICTORS = {"naive": predict_naive, "seasonal": predict_seasonal}


# ---------------------------------------------------------------- calibrated ranges
def calibrate(pred: pd.DataFrame, h: int, months: int = CAL_MONTHS,
              min_rows: int = CAL_MIN_ROWS) -> pd.DataFrame:
    """Shift q10 / q90 per crop so ~10% of outcomes fall below and ~10% above,
    learned from the previous `months` months of errors known before each month.
    pred needs: date, commodity, reported, y_ret, q10, q50, q90."""
    out = pred.copy()
    out["q10_raw"], out["q90_raw"] = pred["q10"], pred["q90"]
    known_on = pred["date"] + pd.Timedelta(days=h)                   # when the outcome was known
    past_ok = pred["reported"].astype(bool) & pred["y_ret"].notna()
    for month in sorted(pred["date"].dt.to_period("M").unique()):
        start = month.to_timestamp()
        window = past_ok & (known_on < start) & (pred["date"] >= start - pd.DateOffset(months=months))
        this = pred["date"].dt.to_period("M") == month
        for crop in pred.loc[this, "commodity"].unique():
            hist = pred[window & (pred["commodity"] == crop)]
            if len(hist) < min_rows:
                continue
            lo = np.quantile(hist["y_ret"] - hist["q10"], 0.10)
            hi = np.quantile(hist["y_ret"] - hist["q90"], 0.90)
            rows = this & (pred["commodity"] == crop)
            out.loc[rows, "q10"] = np.minimum(pred.loc[rows, "q10"] + lo, pred.loc[rows, "q50"])
            out.loc[rows, "q90"] = np.maximum(pred.loc[rows, "q90"] + hi, pred.loc[rows, "q50"])
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(table: pd.DataFrame, models=None, horizons=None, start: str | None = None,
             end: str | None = None, importance: list | None = None) -> pd.DataFrame:
    """Long table of predictions: one row per date, crop, horizon and model."""
    models = models or MODELS
    horizons = horizons or config.HORIZONS
    start = pd.Timestamp(start or config.SPLITS["valid"][0])
    end = pd.Timestamp(end or config.SPLITS["test"][1])
    cal_start = start - pd.DateOffset(months=CAL_MONTHS)
    table = table.sort_values(["date", "commodity"]).reset_index(drop=True)
    out = []
    for h in horizons:
        # xgb also predicts the CAL_MONTHS before `start`, only to calibrate its ranges
        first = cal_start if "xgb" in models else start
        rows = table[(table["date"] >= first) & (table["date"] <= end) & table[f"y_ret_{h}"].notna()]
        for name in models:
            pred = walk_forward_xgb(table, rows, h, importance) if name == "xgb" else PREDICTORS[name](rows, h)
            pred = pred.reindex(rows.index)
            price_now = np.exp(rows["log_price"])
            frame = pd.DataFrame({
                "date": rows["date"], "commodity": rows["commodity"], "split": rows["split"],
                "reported": rows["reported"], "horizon": h, "model": name,
                "price_now": price_now, "y_ret": rows[f"y_ret_{h}"], "y_price": rows[f"y_price_{h}"],
                "q10": pred["q10"], "q50": pred["q50"], "q90": pred["q90"],
                "pred_price": price_now * np.exp(pred["q50"]),
            })
            if name == "xgb":
                frame = calibrate(frame, h).join(pred.filter(regex=r"^(driver|effect)_"))
            out.append(frame[frame["date"] >= start])
    return pd.concat(out, ignore_index=True)


def _scores(g: pd.DataFrame) -> pd.Series:
    err = g["pred_price"] - g["y_price"]
    moved = (g["y_ret"] != 0) & (g["q50"] != 0)
    hit = np.sign(g.loc[moved, "q50"]) == np.sign(g.loc[moved, "y_ret"])

    def coverage(lo, hi):
        has = g[lo].notna()
        inside = (g.loc[has, "y_ret"] >= g.loc[has, lo]) & (g.loc[has, "y_ret"] <= g.loc[has, hi])
        return 100 * inside.mean() if has.any() else np.nan

    return pd.Series({
        "n": len(g),
        "mape": 100 * (err.abs() / g["y_price"]).mean(),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "da": 100 * hit.mean() if len(hit) else np.nan,
        "coverage": coverage("q10", "q90"),
        "coverage_raw": coverage("q10_raw", "q90_raw") if "q10_raw" in g else np.nan,
        "width_pct": 100 * (np.exp(g["q90"]) - np.exp(g["q10"])).mean() if g["q10"].notna().any() else np.nan,
    })


def metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Scores on reported days, per crop and for all crops together."""
    p = preds[preds["reported"].astype(bool)]
    by_crop = p.groupby(["split", "horizon", "model", "commodity"]).apply(_scores, include_groups=False)
    overall = p.groupby(["split", "horizon", "model"]).apply(_scores, include_groups=False)
    overall["commodity"] = "All"
    overall = overall.set_index("commodity", append=True)
    return pd.concat([by_crop, overall]).sort_index().round(2)


# ---------------------------------------------------------------- Diebold-Mariano
def diebold_mariano(d: np.ndarray, h: int) -> tuple[float, float]:
    """DM test on a loss differential series d (model A loss - model B loss), with
    a Newey-West variance over h-1 lags (h-step forecasts overlap) and the
    Harvey-Leybourne-Newbold small-sample correction. Returns (statistic, p-value);
    a negative statistic means model A has the lower loss."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    dc = d - d.mean()
    lags = min(h - 1, T - 1)
    gamma = [dc @ dc / T] + [dc[k:] @ dc[:-k] / T for k in range(1, lags + 1)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var <= 0:
        var = gamma[0] / T
    if var == 0:
        return np.nan, np.nan
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = d.mean() / np.sqrt(var) * correction
    return float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def dm_table(preds: pd.DataFrame, a: str = "xgb", b: str = "naive") -> pd.DataFrame:
    """DM test of absolute % error, model a vs b, per split, horizon and crop.
    'All' uses the daily average loss difference across crops."""
    p = preds[preds["reported"].astype(bool) & preds["model"].isin([a, b])].copy()
    p["ape"] = (p["pred_price"] - p["y_price"]).abs() / p["y_price"]
    wide = p.pivot_table(index=["split", "horizon", "commodity", "date"], columns="model", values="ape")
    wide = wide.dropna(subset=[a, b])
    wide["d"] = wide[a] - wide[b]
    rows = []
    for (split, h), g in wide.groupby(level=["split", "horizon"]):
        groups = [(c, gc["d"]) for c, gc in g.groupby(level="commodity")]
        groups.append(("All", g["d"].groupby(level="date").mean()))
        for crop, d in groups:
            d = d.sort_index(level=-1)
            stat, pval = diebold_mariano(d.to_numpy(), int(h))
            rows.append({"split": split, "horizon": h, "commodity": crop, "n": len(d),
                         "mape_diff_pts": round(100 * d.mean(), 2), "dm_stat": round(stat, 2),
                         "p_value": round(pval, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def results_table(m: pd.DataFrame, split: str, crop: str = "All") -> str:
    """Markdown: one row per model, MAPE and DA per horizon, best value in bold."""
    sub = m.xs((split, crop), level=("split", "commodity")).reset_index()
    horizons = sorted(sub["horizon"].unique())
    lines = ["| Model | " + " | ".join(f"MAPE {h}d | DA {h}d" for h in horizons) + " |",
             "|---|" + "---|---|" * len(horizons)]
    best = {}
    for h in horizons:
        s = sub[sub["horizon"] == h]
        best[(h, "mape")] = s["mape"].min()
        best[(h, "da")] = s["da"].max()
    for model in [x for x in MODELS if x in set(sub["model"])]:
        cells = []
        for h in horizons:
            r = sub[(sub["model"] == model) & (sub["horizon"] == h)].iloc[0]
            for key, fmt in (("mape", "{:.1f}%"), ("da", "{:.1f}%")):
                v = r[key]
                text = "–" if pd.isna(v) else fmt.format(v)
                cells.append(f"**{text}**" if not pd.isna(v) and v == best[(h, key)] else text)
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def coverage_table(m: pd.DataFrame, split: str) -> str:
    sub = m.xs((split, "xgb"), level=("split", "model")).reset_index()
    lines = ["| Crop | Horizon | Coverage raw | Coverage calibrated | Avg range width |",
             "|---|---|---|---|---|"]
    for r in sub.sort_values(["commodity", "horizon"]).itertuples():
        lines.append(f"| {r.commodity} | {r.horizon}d | {r.coverage_raw:.1f}% | {r.coverage:.1f}% "
                     f"| {r.width_pct:.1f}% |")
    return "\n".join(lines)


def dm_markdown(dm: pd.DataFrame, split: str) -> str:
    sub = dm[dm["split"] == split]
    lines = ["| Crop | Horizon | MAPE xgb - naive (pts) | DM stat | p-value |", "|---|---|---|---|---|"]
    for r in sub.sort_values(["commodity", "horizon"]).itertuples():
        mark = " *" if r.p_value < 0.05 else ""
        lines.append(f"| {r.commodity} | {r.horizon}d | {r.mape_diff_pts:+.2f} | {r.dm_stat:.2f} "
                     f"| {r.p_value:.3f}{mark} |")
    return "\n".join(lines)


def importance_table(importance: list) -> pd.DataFrame:
    """Mean |SHAP| per label, per horizon and crop, as % of the total."""
    imp = pd.concat(importance, ignore_index=True)
    imp = imp[imp["date"] >= pd.Timestamp(config.SPLITS["valid"][0])]
    labels = [c for c in imp.columns if c not in ("commodity", "date", "horizon")]
    mean = imp.groupby(["horizon", "commodity"])[labels].mean()
    return (100 * mean.div(mean.sum(axis=1), axis=0)).round(1)


def run(models=None, horizons=None) -> pd.DataFrame:
    path = config.PROCESSED / "features.csv"
    if not path.exists():
        log.info("features.csv not found; building it first")
        features.build()
    table = pd.read_csv(path, parse_dates=["date"])
    importance: list = []
    preds = evaluate(table, models, horizons, importance=importance)
    preds.to_csv(config.PROCESSED / "predictions_m3.csv", index=False)
    m = metrics(preds)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    m.to_csv(config.REPORTS / "m3_metrics.csv")

    has_xgb = "xgb" in set(preds["model"])
    dm = dm_table(preds) if has_xgb and "naive" in set(preds["model"]) else pd.DataFrame()
    if not dm.empty:
        dm.to_csv(config.REPORTS / "m3_dm_test.csv", index=False)
    if importance:
        imp = importance_table(importance)
        imp.to_csv(config.REPORTS / "m3_shap_importance.csv")

    sections = []
    for split in [s for s in ("valid", "test") if s in m.index.get_level_values("split")]:
        span = f"{config.SPLITS[split][0][:4]}-{config.SPLITS[split][1][:4]}"
        sections.append(f"## {split} ({span}), all crops\n\n" + results_table(m, split))
        if has_xgb:
            sections.append(f"### {split}: 10-90 range coverage (xgb)\n\n" + coverage_table(m, split))
        if not dm.empty:
            sections.append(f"### {split}: Diebold-Mariano, xgb vs naive (absolute % error)\n\n"
                            "Negative = xgb better; * = significant at 5%.\n\n" + dm_markdown(dm, split))
        for crop in sorted(c for c in m.index.get_level_values("commodity").unique() if c != "All"):
            sections.append(f"### {split}: {crop}\n\n" + results_table(m, split, crop))
    if importance:
        top = imp.groupby(level="horizon").mean().T
        lines = ["| Driver | " + " | ".join(f"{h}d" for h in top.columns) + " |",
                 "|---|" + "---|" * len(top.columns)]
        for label, r in top.sort_values(top.columns[0], ascending=False).head(10).iterrows():
            lines.append(f"| {label} | " + " | ".join(f"{v:.1f}%" for v in r) + " |")
        sections.append("## What drives the xgb forecasts (share of mean |SHAP|, all crops)\n\n"
                        + "\n".join(lines))
    text = ("# M3 results: walk-forward, reported days only\n\n"
            "MAPE: lower is better. DA: directional accuracy, higher is better (naive has none).\n\n"
            + "\n\n".join(sections) + "\n")
    (config.REPORTS / "m3_results.md").write_text(text, encoding="utf-8")
    print(text)
    return m
