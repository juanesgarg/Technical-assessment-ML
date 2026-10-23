"""Realized and unrealized P&L, net of commissions, per account and session.

Method: average cost per (account, instrument), walking the fills in time order.
- A fill that opens or adds to the position updates the average price.
- A fill that reduces or closes it realizes P&L = closed qty * (price - avg) * direction * point_value.
- A fill that flips it (long to short or back) closes everything and opens the rest at the fill price.
point_value comes from instruments.csv: 1 NQ point is worth $20, 1 MNQ point $2.
"""
import numpy as np
import pandas as pd

from .config import CUT


def add_realized_pnl(fills: pd.DataFrame, instruments: pd.DataFrame) -> pd.DataFrame:
    """Add per fill: signed_qty, position_after, avg_price_after and realized_gross_usd."""
    pv = instruments.set_index("symbol")["point_value_usd"]
    out = fills.sort_values(["filled_at", "id"]).copy()
    out["point_value_usd"] = out["instrument_symbol"].map(pv)
    out["signed_qty"] = np.where(out["side"] == "BUY", 1, -1) * out["quantity"]

    pos_after, avg_after, realized = [], [], []
    state = {}  # (account, symbol) -> (position, average price)
    for acc, sym, q, px, point_value in out[["account_id", "instrument_symbol", "signed_qty", "price",
                                             "point_value_usd"]].itertuples(index=False):
        pos, avg = state.get((acc, sym), (0, 0.0))
        pnl = 0.0
        if pos == 0 or np.sign(pos) == np.sign(q):          # opens or adds
            avg = (avg * abs(pos) + px * abs(q)) / (abs(pos) + abs(q))
            pos += q
        else:                                               # reduces, closes or flips
            closed = min(abs(q), abs(pos))
            pnl = closed * (px - avg) * np.sign(pos) * point_value
            pos += q
            if pos == 0:
                avg = 0.0
            elif np.sign(pos) == np.sign(q):                # flipped: the remainder opens at px
                avg = px
        state[(acc, sym)] = (pos, avg)
        pos_after.append(pos)
        avg_after.append(avg)
        realized.append(pnl)

    out["position_after"] = pos_after
    out["avg_price_after"] = avg_after
    out["realized_gross_usd"] = realized
    return out


def open_positions(fills_pnl: pd.DataFrame, market_prices: pd.DataFrame) -> pd.DataFrame:
    """Positions open at the cut, valued at the market_prices.csv mark."""
    last = fills_pnl.groupby(["account_id", "instrument_symbol"]).tail(1)
    op = last.loc[last["position_after"] != 0,
                  ["account_id", "instrument_symbol", "session_date", "position_after", "avg_price_after",
                   "point_value_usd"]].rename(columns={"position_after": "position", "avg_price_after": "avg_price"})
    marks = market_prices.set_index("symbol")
    op["mark_price"] = op["instrument_symbol"].map(marks["mark_price"])
    assert (op["instrument_symbol"].map(marks["as_of"]) == CUT).all(), "marks are not as of the cut"
    op["unrealized_usd"] = (op["mark_price"] - op["avg_price"]) * op["position"] * op["point_value_usd"]
    return op.reset_index(drop=True)


def session_pnl(fills_pnl: pd.DataFrame, open_pos: pd.DataFrame, accounts: pd.DataFrame,
                traders: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, session): activity, realized P&L, commissions and unrealized P&L."""
    g = fills_pnl.groupby(["account_id", "session_idx", "session_date"])
    sp = g.agg(
        n_fills=("id", "size"),
        n_orders=("order_id", "nunique"),          # partial fills of one order count once
        contracts=("quantity", "sum"),
        instruments=("instrument_symbol", lambda s: ",".join(sorted(set(s)))),
        max_abs_position=("position_after", lambda s: s.abs().max()),
        realized_gross_usd=("realized_gross_usd", "sum"),
        commissions_usd=("commission_usd", "sum"),
        first_fill=("filled_at", "min"),
        last_fill=("filled_at", "max"),
    ).reset_index()

    # Every position open at the cut was opened in today's session (earlier sessions ended flat,
    # see quality.py), so its unrealized P&L belongs to the session it is open in.
    unreal = open_pos.groupby(["account_id", "session_date"])["unrealized_usd"].sum()
    sp["unrealized_usd"] = sp.set_index(["account_id", "session_date"]).index.map(unreal).fillna(0.0)

    sp["net_realized_usd"] = sp["realized_gross_usd"] - sp["commissions_usd"]
    sp["net_pnl_usd"] = sp["net_realized_usd"] + sp["unrealized_usd"]

    who = accounts[["id", "trader_id", "account_type", "status", "balance", "max_position_size"]].merge(
        traders[["id", "broker_id", "first_name", "last_name"]], left_on="trader_id", right_on="id",
        suffixes=("", "_t")).drop(columns="id_t")
    sp = sp.merge(who, left_on="account_id", right_on="id").drop(columns="id")
    cols = ["trader_id", "first_name", "last_name", "broker_id", "account_id", "account_type", "status",
            "session_idx", "session_date"]
    return sp[cols + [c for c in sp.columns if c not in cols]].sort_values(["account_id", "session_idx"])
