"""Task 1: sessions that are unusually bad for each trader.

1. Sum each person's (trader_id) accounts per session; the demo account is excluded.
2. Per-trader baseline using ONLY earlier sessions (no look-ahead):
   median net P&L, robust spread (MAD x 1.4826), median contracts and worst prior session.
3. z = (session P&L - median) / spread. Today's session is ranked, losses only.
"""
import numpy as np
import pandas as pd

MIN_HIST = 5        # minimum prior sessions before we trust a trader's "normal"
MIN_SPREAD_USD = 50  # spread floor: avoids huge z for traders who barely move


def trader_sessions(sess: pd.DataFrame) -> pd.DataFrame:
    """From (account, session) to (trader, session)."""
    real = sess[sess["account_type"] != "demo"]
    return (real.groupby(["trader_id", "session_date"])
            .agg(n_accounts=("account_id", "nunique"), n_orders=("n_orders", "sum"),
                 contracts=("contracts", "sum"), net_pnl_usd=("net_pnl_usd", "sum"))
            .reset_index())


def _baseline_one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("session_date").copy()
    rows = []
    for i in range(len(g)):
        h = g.iloc[:i]                                   # only sessions before session i
        if len(h) == 0:
            rows.append((0, np.nan, np.nan, np.nan, np.nan))
            continue
        m = h["net_pnl_usd"].median()
        rows.append((len(h), m, 1.4826 * (h["net_pnl_usd"] - m).abs().median(),
                     h["contracts"].median(), h["net_pnl_usd"].min()))
    base = pd.DataFrame(rows, columns=["n_prior", "base_pnl", "base_spread", "base_contracts", "worst_prior"],
                        index=g.index)
    return g.join(base)


def add_baseline(ts: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([_baseline_one(g) for _, g in ts.groupby("trader_id")], ignore_index=True)
    out.loc[out["n_prior"] < MIN_HIST, ["base_pnl", "base_spread"]] = np.nan
    return out


def score_session(tb: pd.DataFrame, session_date) -> pd.DataFrame:
    """z and evidence for one session. Returns every trader who traded that day."""
    day = tb[tb["session_date"] == pd.Timestamp(session_date)].copy()
    day["z"] = (day["net_pnl_usd"] - day["base_pnl"]) / day["base_spread"].clip(lower=MIN_SPREAD_USD)
    day["size_vs_normal"] = day["contracts"] / day["base_contracts"]
    day["worse_than_worst_prior"] = day["net_pnl_usd"] < day["worst_prior"]
    return day


def top_n(day: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the n worst traders with enough history, traders without enough history)."""
    ranked = day[day["base_pnl"].notna() & (day["net_pnl_usd"] < 0)].nsmallest(n, "z").copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked, day[day["base_pnl"].isna()]
