"""Tests for the hand-made reference tables (MSP, calendar, crop seasons)."""
from __future__ import annotations

import pandas as pd
import pytest

from krishibhav import config, reference


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_committed_reference_files_are_valid():
    msp, calendar, seasons = reference.load_msp(), reference.load_calendar(), reference.load_seasons()
    assert {"Groundnut", "Bajra"} <= set(msp["crop"])
    assert (calendar["name"] == "Diwali").sum() >= 12
    assert set(seasons["crop"]) == set(config.COMMODITIES)


def test_msp_season_boundaries(tmp_path):
    msp = reference.load_msp(write(tmp_path / "msp.csv", """# comment line
crop,season,msp_rs_qtl,status,source
Bajra,2024-25,2625,verified,
Bajra,2025-26,2775,verified,
"""))
    s = reference.msp_on(["2024-09-30", "2024-10-01", "2025-09-30", "2025-10-01", "2026-10-01"], "Bajra", msp)
    assert pd.isna(s.iloc[0])                                     # before the first season
    assert s.tolist()[1:4] == [2625.0, 2625.0, 2775.0]            # seasons switch on 1 Oct
    assert pd.isna(s.iloc[4])                                     # season not entered yet
    assert reference.msp_on(["2025-01-01"], "Potato", msp).isna().all()   # no MSP for potato


def test_msp_problems_are_all_listed(tmp_path):
    path = write(tmp_path / "msp.csv", """crop,season,msp_rs_qtl,status,source
Bajra,2025-2026,2775,verified,
Bajra,2024-25,-5,verified,
Groundnut,2024-25,6783,maybe,
Groundnut,2024-25,6783,verified,
""")
    with pytest.raises(ValueError) as err:
        reference.load_msp(path)
    msg = str(err.value)
    assert "4 problem(s)" in msg
    assert "season must look like 2025-26" in msg and "positive number" in msg
    assert "status must be one of" in msg and "listed more than once" in msg


def test_calendar_problems_are_all_listed(tmp_path):
    path = write(tmp_path / "calendar.csv", """start,end,kind,name,crop,market,status
2025-10-20,2025-10-18,festival,Diwali,,,verified
2025-13-01,,holiday,Bad date,,,verified
2025-01-26,,apmc_holiday,,,,verified
""")
    with pytest.raises(ValueError) as err:
        reference.load_calendar(path)
    msg = str(err.value)
    assert "is before start" in msg and "start must be YYYY-MM-DD" in msg
    assert "kind must be one of" in msg and "name is empty" in msg


def test_seasons_reject_bad_month_day(tmp_path):
    path = write(tmp_path / "crop_seasons.csv", """crop,phase,start,end,status,note
Potato,harvest,02-30,03-31,check,
Potato,Harvest Time,01-15,13-01,check,
""")
    with pytest.raises(ValueError) as err:
        reference.load_seasons(path)
    msg = str(err.value)
    assert "start must be MM-DD, got '02-30'" in msg and "end must be MM-DD, got '13-01'" in msg
    assert "phase must be lowercase" in msg


def test_daily_flags(tmp_path):
    calendar = reference.load_calendar(write(tmp_path / "calendar.csv", """start,end,kind,name,crop,market,status
2025-10-20,,festival,Diwali,,,verified
2026-11-08,,festival,Diwali,,,verified
2025-10-18,2025-10-25,apmc_holiday,Diwali vacation,,,check
2025-12-01,2026-01-31,policy,Castor duty change,Castor Seed,,check
2025-12-10,,apmc_holiday,Deesa strike,,Deesa,check
"""))
    seasons = reference.load_seasons(write(tmp_path / "crop_seasons.csv", """crop,phase,start,end,status,note
Castor Seed,harvest,12-01,04-30,check,
Castor Seed,sowing,07-01,08-31,check,
Bajra,harvest,09-15,10-31,check,
"""))
    msp = reference.load_msp(write(tmp_path / "msp.csv", """crop,season,msp_rs_qtl,status,source
Bajra,2025-26,2775,verified,
"""))
    dates = ["2025-10-19", "2025-12-10", "2026-01-15", "2026-06-01"]

    castor = reference.daily_flags(dates, "Castor Seed", market="Deesa",
                                   msp=msp, calendar=calendar, seasons=seasons)
    assert castor["apmc_holiday"].tolist() == [True, True, False, False]   # vacation, Deesa strike
    assert castor["policy"].tolist() == [False, True, True, False]
    assert castor["in_harvest"].tolist() == [False, True, True, False]     # Dec-Apr wraps the year
    assert not castor["in_sowing"].any()
    assert castor["days_to_diwali"].tolist() == [1, 333, 297, 160]
    assert castor["msp_rs_qtl"].isna().all()

    bajra = reference.daily_flags(dates, "Bajra", market="Palanpur",
                                  msp=msp, calendar=calendar, seasons=seasons)
    assert bajra["apmc_holiday"].tolist() == [True, False, False, False]   # Deesa strike not here
    assert not bajra["policy"].any()                                       # castor-only event
    assert bajra["in_harvest"].tolist() == [True, False, False, False]
    assert bajra["msp_rs_qtl"].tolist() == [2775.0] * 4
    assert "in_sowing" not in bajra.columns


def test_closed_day_candidates(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_START", "2026-01-01")
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-07", "2026-01-08", "2026-01-06"]),
        "market": ["Deesa", "Palanpur", "Deesa", "Tharad"],              # Tharad not configured
        "modal_price": [1000.0, 1000.0, 1000.0, 1000.0],
    })
    closed = reference.closed_day_candidates(prices)
    assert [str(d) for d in closed["date"]] == ["2026-01-06"]             # Mon-Thu, Tue missing
