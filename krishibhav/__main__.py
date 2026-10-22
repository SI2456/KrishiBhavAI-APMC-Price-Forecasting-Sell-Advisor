"""Command line: python -m krishibhav <command>

  init-db         create the database tables
  load-files      clean + load every file in data/raw/agmarknet
  fetch-today     pull today's Gujarat prices from data.gov.in (run every evening)
  load-datagov    re-load all saved data.gov.in responses
  fetch-weather   weather history (incremental) + 16-day forecast from Open-Meteo
  load-weather    re-load all saved forecast files
  build-series    write data/processed/daily_series.csv
  build-features  write data/processed/features.csv (model inputs + targets)
  evaluate        M3: naive, seasonal-naive and XGBoost, walk-forward on 2023-2025
  backtest        M5: sell/hold advisor replayed on 2023-2025 vs selling on arrival
  report          coverage table + price charts in reports/
  check-markets   how often each market reports each crop, per year (Agmarknet history)
  check-reference validate data/reference/*.csv and suggest APMC holidays
  daily           fetch-today + fetch-weather + build-series (for the scheduler)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="krishibhav", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    p = sub.add_parser("load-files")
    p.add_argument("--folder", type=Path, default=None)
    p.add_argument("--keep-all", action="store_true",
                   help="keep every commodity and market, not just those in config.py")
    p = sub.add_parser("fetch-today")
    p.add_argument("--raw-only", action="store_true", help="save the JSON but skip the database")
    sub.add_parser("load-datagov")
    p = sub.add_parser("fetch-weather")
    p.add_argument("--forecast-only", action="store_true")
    p.add_argument("--raw-only", action="store_true",
                   help="only save today's forecast JSON; skip history and the database")
    sub.add_parser("load-weather")
    sub.add_parser("build-series")
    sub.add_parser("build-features")
    p = sub.add_parser("evaluate")
    p.add_argument("--models", default="naive,seasonal,xgb")
    p.add_argument("--horizons", default="", help="e.g. 7,14 (default: config.HORIZONS)")
    p = sub.add_parser("backtest")
    p.add_argument("--lot", type=float, default=50.0, help="lot size in quintal for the example advice")
    sub.add_parser("report")
    p = sub.add_parser("check-markets")
    p.add_argument("--markets", default="",
                   help='extra Agmarknet market names, comma separated, e.g. "Tharad,Diyodar"')
    p.add_argument("--refresh", action="store_true", help="download again instead of using saved files")
    sub.add_parser("check-reference")
    sub.add_parser("daily")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from . import db, ingest, datagov, weather, series, report, markets, reference, features, models, advisor

    if args.cmd == "init-db":
        db.init_db(db.get_engine())
        print("Tables ready.")
    elif args.cmd == "load-files":
        ingest.load_files(args.folder, keep_all=args.keep_all)
    elif args.cmd == "fetch-today":
        datagov.fetch_today(load=not args.raw_only)
    elif args.cmd == "load-datagov":
        ingest.load_datagov_raw()
    elif args.cmd == "fetch-weather":
        weather.fetch_weather(history=not (args.forecast_only or args.raw_only), load=not args.raw_only)
    elif args.cmd == "load-weather":
        weather.load_forecasts_raw()
    elif args.cmd == "build-series":
        series.build_series()
    elif args.cmd == "build-features":
        features.build()
    elif args.cmd == "evaluate":
        horizons = [int(h) for h in args.horizons.split(",") if h.strip()] or None
        models.run([m.strip() for m in args.models.split(",") if m.strip()], horizons)
    elif args.cmd == "backtest":
        advisor.run(args.lot)
    elif args.cmd == "report":
        report.run_report()
    elif args.cmd == "check-markets":
        extra = [m.strip() for m in args.markets.split(",") if m.strip()]
        markets.check_markets(extra, refresh=args.refresh)
    elif args.cmd == "check-reference":
        reference.check_reference()
    elif args.cmd == "daily":
        failed = False
        for step in (datagov.fetch_today, lambda: weather.fetch_weather(), series.build_series):
            try:
                step()
            except Exception as exc:          # one failing source should not stop the rest
                logging.exception("Step failed: %s", exc)
                failed = True
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
