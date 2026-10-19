"""Data quality report: coverage per series and a price chart for each one.

Run it after every load. Drop any market/commodity with coverage under ~70%
from modelling; note the decision in your report.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config
from .series import read_prices, combine_daily

log = logging.getLogger(__name__)


def coverage_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, commodity), s in daily.groupby(["market", "commodity"]):
        dates = pd.DatetimeIndex(sorted(s["date"].unique()))
        all_days = pd.date_range(dates.min(), dates.max(), freq="D")
        trading_days = all_days[all_days.dayofweek != 6]          # APMCs mostly shut on Sunday
        gaps = dates.to_series().diff().dt.days.fillna(0)
        rows.append({
            "market": market,
            "commodity": commodity,
            "first": dates.min().date(),
            "last": dates.max().date(),
            "days_reported": len(dates),
            "coverage_pct": round(100 * len(dates) / max(len(trading_days), 1), 1),
            "longest_gap_days": int(gaps.max()),
            "median_modal_rs_qtl": round(float(s["modal_price"].median()), 0),
            "arrivals_known_pct": round(100 * s["arrivals_tonnes"].notna().mean(), 1),
        })
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
    daily = combine_daily(prices)
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
