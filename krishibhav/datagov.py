"""Fetch today's Gujarat mandi prices from the data.gov.in API.

This API only returns the latest reported day, so run it every evening.
Each response is saved to data/raw/datagov/<date>.json before loading, so the
history can always be rebuilt with `python -m krishibhav load-datagov`.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date

import requests

from . import config, db
from .clean import normalise, to_records

log = logging.getLogger(__name__)
API = "https://api.data.gov.in/resource/{rid}"
PAGE = 1000
# data.gov.in has used both spellings for field filters; try them in order.
STATE_FILTERS = ["filters[state.keyword]", "filters[state]"]


def _get(session: requests.Session, url: str, params: dict, tries: int = 4) -> dict:
    for attempt in range(1, tries + 1):
        try:
            resp = session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == tries:
                raise
            wait = 5 * attempt
            log.warning("data.gov.in attempt %d failed (%s); retrying in %ds", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_state(api_key: str, state: str = config.STATE, session: requests.Session | None = None
                ) -> list[dict]:
    if not api_key:
        raise RuntimeError("Set DATAGOV_API_KEY in .env (free key from data.gov.in → My Account).")
    session = session or requests.Session()
    url = API.format(rid=config.DATAGOV_RESOURCE_ID)

    for filter_key in STATE_FILTERS:
        records: list[dict] = []
        offset = 0
        while True:
            params = {"api-key": api_key, "format": "json", "limit": PAGE,
                      "offset": offset, filter_key: state}
            payload = _get(session, url, params)
            batch = payload.get("records", []) or []
            records.extend(batch)
            total = int(payload.get("total", 0) or 0)
            offset += PAGE
            if not batch or offset >= total:
                break
        if records:
            log.info("data.gov.in returned %d %s records (filter %s)", len(records), state, filter_key)
            return records
    log.warning("data.gov.in returned no records for %s. The market may not have reported yet "
                "(try after 8 pm), or the resource/filter names have changed.", state)
    return []


def fetch_today(save_raw: bool = True, load: bool = True) -> int:
    records = fetch_state(config.DATAGOV_API_KEY)
    if not records:
        return 0
    if save_raw:
        config.RAW_DATAGOV.mkdir(parents=True, exist_ok=True)
        out = config.RAW_DATAGOV / f"{date.today().isoformat()}.json"
        out.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("Saved raw response to %s", out)
    if not load:
        return len(records)

    import pandas as pd
    df, rep = normalise(pd.DataFrame(records), source="datagov", file_hint="datagov")
    engine = db.get_engine()
    db.init_db(engine)
    n = db.upsert(engine, db.mandi_prices, to_records(df, db.now_utc()), db.PRICE_KEY)
    log.info(rep.line())
    return n
