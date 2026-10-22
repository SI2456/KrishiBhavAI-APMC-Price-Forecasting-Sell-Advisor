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

Metrics per model, horizon, split and crop:
  MAPE %       mean |predicted - actual| / actual price
  RMSE         Rs/quintal
  DA %         directional accuracy: did it call up vs down (days with no
               change, and the naive model, are left out)
  coverage %   share of actual returns inside the 10-90 range (xgb only; ~80 is honest)
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


def walk_forward_xgb(table: pd.DataFrame, rows: pd.DataFrame, h: int) -> pd.DataFrame:
    columns = feature_columns(table)
    months = sorted(rows["date"].dt.to_period("M").unique())
    parts = []
    for month in months:
        start = month.to_timestamp()
        target = rows[rows["date"].dt.to_period("M") == month]
        train = training_rows(table, start, h)
        model = fit_xgb(train, h, columns)
        parts.append(predict_xgb_model(model, target, columns))
        log.info("xgb h=%d %s: trained on %d rows up to %s, predicted %d",
                 h, month, len(train), train["date"].max().date(), len(target))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["q10", "q50", "q90"])


PREDICTORS = {"naive": predict_naive, "seasonal": predict_seasonal}


# ---------------------------------------------------------------- evaluation
def evaluate(table: pd.DataFrame, models=None, horizons=None,
             start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Long table of predictions: one row per date, crop, horizon and model."""
    models = models or MODELS
    horizons = horizons or config.HORIZONS
    start = pd.Timestamp(start or config.SPLITS["valid"][0])
    end = pd.Timestamp(end or config.SPLITS["test"][1])
    table = table.sort_values(["date", "commodity"]).reset_index(drop=True)
    out = []
    for h in horizons:
        rows = table[(table["date"] >= start) & (table["date"] <= end) & table[f"y_ret_{h}"].notna()]
        for name in models:
            pred = walk_forward_xgb(table, rows, h) if name == "xgb" else PREDICTORS[name](rows, h)
            pred = pred.reindex(rows.index)
            price_now = np.exp(rows["log_price"])
            out.append(pd.DataFrame({
                "date": rows["date"], "commodity": rows["commodity"], "split": rows["split"],
                "reported": rows["reported"], "horizon": h, "model": name,
                "price_now": price_now, "y_ret": rows[f"y_ret_{h}"], "y_price": rows[f"y_price_{h}"],
                "q10": pred["q10"], "q50": pred["q50"], "q90": pred["q90"],
                "pred_price": price_now * np.exp(pred["q50"]),
            }))
    return pd.concat(out, ignore_index=True)


def _scores(g: pd.DataFrame) -> pd.Series:
    err = g["pred_price"] - g["y_price"]
    moved = (g["y_ret"] != 0) & (g["q50"] != 0)
    hit = np.sign(g.loc[moved, "q50"]) == np.sign(g.loc[moved, "y_ret"])
    has_range = g["q10"].notna()
    inside = (g.loc[has_range, "y_ret"] >= g.loc[has_range, "q10"]) & \
             (g.loc[has_range, "y_ret"] <= g.loc[has_range, "q90"])
    return pd.Series({
        "n": len(g),
        "mape": 100 * (err.abs() / g["y_price"]).mean(),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "da": 100 * hit.mean() if len(hit) else np.nan,
        "coverage": 100 * inside.mean() if len(inside) else np.nan,
    })


def metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Scores on reported days, per crop and for all crops together."""
    p = preds[preds["reported"].astype(bool)]
    by_crop = p.groupby(["split", "horizon", "model", "commodity"]).apply(_scores, include_groups=False)
    overall = p.groupby(["split", "horizon", "model"]).apply(_scores, include_groups=False)
    overall["commodity"] = "All"
    overall = overall.set_index("commodity", append=True)
    return pd.concat([by_crop, overall]).sort_index().round(2)


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


def run(models=None, horizons=None) -> pd.DataFrame:
    path = config.PROCESSED / "features.csv"
    if not path.exists():
        log.info("features.csv not found; building it first")
        features.build()
    table = pd.read_csv(path, parse_dates=["date"])
    preds = evaluate(table, models, horizons)
    preds.to_csv(config.PROCESSED / "predictions_m3.csv", index=False)
    m = metrics(preds)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    m.to_csv(config.REPORTS / "m3_metrics.csv")

    sections = []
    for split in [s for s in ("valid", "test") if s in m.index.get_level_values("split")]:
        sections.append(f"## {split} ({config.SPLITS[split][0][:4]}-{config.SPLITS[split][1][:4]}), all crops\n\n"
                        + results_table(m, split))
        for crop in sorted(c for c in m.index.get_level_values("commodity").unique() if c != "All"):
            sections.append(f"### {split}: {crop}\n\n" + results_table(m, split, crop))
    text = ("# M3 results: walk-forward, reported days only\n\n"
            "MAPE: lower is better. DA: directional accuracy, higher is better (naive has none).\n\n"
            + "\n\n".join(sections) + "\n")
    (config.REPORTS / "m3_results.md").write_text(text, encoding="utf-8")
    print(text)
    return m
