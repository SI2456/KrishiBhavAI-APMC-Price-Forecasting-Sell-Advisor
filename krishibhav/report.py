"""Data quality report: coverage per series and a price chart for each one.

Run it after every load. Drop any market/commodity with coverage under ~70%
from modelling; note the decision in your report. Coverage is measured over
the whole history window (config.HISTORY_START to the latest date in the data),
so a series that only starts in 2022 shows low coverage instead of 100%.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config
from .series import daily_prices, read_prices

log = logging.getLogger(__name__)


def coverage_table(daily: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    window_start = pd.Timestamp(start or config.HISTORY_START)
    window_end = daily["date"].max()
    all_days = pd.date_range(window_start, window_end, freq="D")
    trading_days = all_days[all_days.dayofweek != 6]              # APMCs mostly shut on Sunday
    rows = []
    for (market, commodity), s in daily.groupby(["market", "commodity"]):
        priced = s[s["modal_price"].notna()]
        dates = pd.DatetimeIndex(sorted(priced["date"].unique()))
        if dates.empty:
            continue
        in_window = dates[(dates >= window_start) & (dates.dayofweek != 6)]
        edges = pd.DatetimeIndex([window_start - pd.Timedelta(days=1)]).append(
            dates[dates >= window_start]).append(
            pd.DatetimeIndex([window_end + pd.Timedelta(days=1)]))
        gaps = edges.to_series().diff().dt.days.fillna(0) - 1     # empty days between reports
        rows.append({
            "market": market,
            "commodity": commodity,
            "variety": s["variety"].iloc[0] if "variety" in s.columns else "",
            "first": dates.min().date(),
            "last": dates.max().date(),
            "days_reported": len(dates),
            "coverage_pct": round(100 * len(in_window) / max(len(trading_days), 1), 1),
            "longest_gap_days": int(gaps.max()),
            "median_modal_rs_qtl": round(float(s["modal_price"].median()), 0),
            "arrivals_known_pct": round(100 * s["arrivals_tonnes"].notna().mean(), 1),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["market", "commodity"]).reset_index(drop=True)


def plot_series(daily: pd.DataFrame) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = config.REPORTS / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for (market, commodity), s in daily.groupby(["market", "commodity"]):
        s = s.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.fill_between(s["date"], s["min_price"], s["max_price"], alpha=0.2, linewidth=0, label="min to max")
        ax.plot(s["date"], s["modal_price"], linewidth=1, label="modal")
        ax.set_title(f"{commodity} at {market}")
        ax.set_ylabel("Rs / quintal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        name = f"{market}_{commodity}".lower().replace(" ", "-")
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)
        n += 1
    return n


def run_report() -> pd.DataFrame:
    prices = read_prices()
    if prices.empty:
        log.warning("No prices in the database yet.")
        return pd.DataFrame()
    daily = daily_prices(prices)
    table = coverage_table(daily)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS / "coverage.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.to_string(index=False))
    print("\nRows by source:")
    print(prices["source"].value_counts().to_string())
    n = plot_series(daily)
    log.info("Saved coverage.csv and %d charts in %s", n, config.REPORTS)
    return table
