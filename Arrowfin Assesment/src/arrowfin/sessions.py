"""Assign each fill to its CME trading session."""
import pandas as pd

from .config import CUT, EXCHANGE_TZ, SESSION_OPEN_HOUR


def add_session(fills: pd.DataFrame) -> pd.DataFrame:
    """Add `session_date` (CME trade date) and `session_idx` (1 = first session).

    Trick: convert to Chicago time and add 7 hours. 17:00 CT lands on 00:00 of the next day and
    16:00 CT on 23:00 of the same day, so the resulting date is the trade date.
    Using the time zone (not a fixed UTC-5/UTC-6 offset) handles daylight saving automatically.
    """
    out = fills.copy()
    local = out["filled_at"].dt.tz_convert(EXCHANGE_TZ).dt.tz_localize(None)
    out["filled_at_ct"] = local
    out["session_date"] = (local + pd.Timedelta(hours=24 - SESSION_OPEN_HOUR)).dt.normalize()

    # Check: the market is closed 16:00-17:00 CT, so there should be no fills.
    in_break = local.dt.hour == 16
    assert not in_break.any(), f"{in_break.sum()} fills in the 16:00-17:00 CT daily break"
    # Monday sessions open on Sunday evening, so no trade date should fall on a weekend.
    assert (out["session_date"].dt.dayofweek < 5).all(), "session with a weekend trade date"

    sessions = sorted(out["session_date"].unique())
    out["session_idx"] = out["session_date"].map({d: i + 1 for i, d in enumerate(sessions)})
    return out


def session_calendar(fills_with_session: pd.DataFrame) -> pd.DataFrame:
    """One row per session with its UTC window and whether it is complete at the cut."""
    cal = (fills_with_session.groupby(["session_idx", "session_date"])
           .agg(n_fills=("id", "size"), first_fill=("filled_at", "min"), last_fill=("filled_at", "max"))
           .reset_index())
    open_ct = cal["session_date"] - pd.Timedelta(hours=24 - SESSION_OPEN_HOUR)
    cal["opens_utc"] = open_ct.dt.tz_localize(EXCHANGE_TZ).dt.tz_convert("UTC")
    cal["closes_utc"] = (cal["session_date"] + pd.Timedelta(hours=16)).dt.tz_localize(EXCHANGE_TZ).dt.tz_convert("UTC")
    cal["is_complete"] = cal["closes_utc"] <= CUT
    cal["hours_elapsed_at_cut"] = ((cal["closes_utc"].clip(upper=CUT) - cal["opens_utc"]).dt.total_seconds() / 3600).round(1)
    return cal
