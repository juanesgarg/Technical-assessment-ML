"""Data quality report: values that are valid but change how the numbers must be read.

Nothing is dropped or fixed silently. Each finding goes into a table with what it is, who it
affects and what it implies for the analysis, so it can be cited in the README.
"""
import pandas as pd

from .config import CUT


def _row(check, severity, entity, detail, implication):
    return {"check": check, "severity": severity, "entity": entity, "detail": detail, "implication": implication}


def quality_report(t: dict[str, pd.DataFrame], fills_pnl: pd.DataFrame, calendar: pd.DataFrame,
                   open_pos: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    a, tr = t["accounts"], t["traders"]
    f = fills_pnl
    rows = []

    # 1. Today's session is incomplete.
    for r in calendar[~calendar["is_complete"]].itertuples():
        rows.append(_row("partial_session", "high", str(r.session_date.date()),
                         f"Session {r.session_date.date()} is {r.hours_elapsed_at_cut} of 23 h old at the cut {CUT}.",
                         "Do not compare today's P&L with complete sessions without adjusting for it."))

    # 2. Fills whose UTC date differs from their session (the calendar-day trap).
    utc_date = f["filled_at"].dt.tz_localize(None).dt.normalize()
    mism = f[utc_date != f["session_date"]]
    if len(mism):
        rows.append(_row("utc_date_differs_from_session", "medium", ",".join(sorted(mism["account_id"].unique())),
                         f"{len(mism)} fills whose UTC date differs from their CME session "
                         f"(evening open from {mism['filled_at'].min()}).",
                         "Grouping by UTC day would put them in the wrong session."))

    # 3. Non-active accounts with fills.
    for r in a[a["status"] != "active"].itertuples():
        fa = f[f["account_id"] == r.id]
        if len(fa):
            rows.append(_row("inactive_account_with_fills", "high", f"{r.id} ({r.trader_id})",
                             f"status={r.status}, {len(fa)} fills between {fa['session_date'].min().date()} and "
                             f"{fa['session_date'].max().date()}, buying_power={r.buying_power}.",
                             "With no status-change date we cannot tell whether it traded while already closed/restricted."))

    # 4. Accounts without fills, demo accounts, zero-balance accounts.
    for r in a[~a["id"].isin(f["account_id"])].itertuples():
        rows.append(_row("account_without_fills", "low", f"{r.id} ({r.trader_id})",
                         f"account_type={r.account_type}, status={r.status}.", "Appears in no ranking."))
    for r in a[a["account_type"] == "demo"].itertuples():
        rows.append(_row("demo_account", "medium", f"{r.id} ({r.trader_id})", "Demo account: not real money.",
                         "Exclude from any risk ranking."))
    for r in a[(a["balance"] <= 0) & a["id"].isin(f["account_id"])].itertuples():
        rows.append(_row("zero_balance_with_fills", "high", f"{r.id} ({r.trader_id})",
                         f"balance={r.balance}, buying_power={r.buying_power}, yet it trades.",
                         "Cannot normalise by balance; possibly a liquidated account or missing data."))

    # 5. Short history: a trader's own 'normal' cannot be estimated from 1-2 sessions.
    n_sess = sess.groupby("account_id")["session_idx"].nunique()
    for acc, n in n_sess[n_sess <= 3].items():
        rows.append(_row("short_history", "high", acc, f"Only {n} session(s) with fills.",
                         "Its own baseline is unreliable: use a peer baseline or list it separately."))

    # 6. Traders with several accounts, and duplicate names across different traders.
    multi = a.groupby("trader_id")["id"].agg(list)
    for trader, accs in multi[multi.str.len() > 1].items():
        rows.append(_row("multi_account_trader", "medium", trader, f"Accounts: {', '.join(accs)}.",
                         "Decide whether 'trader' means person (sum accounts) or account; the ranking asks for traders."))
    full = tr["first_name"] + " " + tr["last_name"]
    for name, ids in tr.groupby(full)["id"].agg(list).items():
        if len(ids) > 1:
            brokers = tr.set_index("id").loc[ids, "broker_id"].tolist()
            rows.append(_row("duplicate_name", "medium", ", ".join(ids),
                             f"'{name}' appears as {len(ids)} traders (brokers {', '.join(brokers)}).",
                             "Identify by trader_id, never by name; we do not know if it is the same person."))

    # 7. Accounts of the same trader trading in lockstep (same symbol, same side, minutes apart).
    for trader, accs in multi[multi.str.len() > 1].items():
        sub = f[f["account_id"].isin(accs)].sort_values("filled_at")
        if sub["account_id"].nunique() < 2:
            continue
        sub = sub.assign(prev_acc=sub["account_id"].shift(), prev_sym=sub["instrument_symbol"].shift(),
                         prev_side=sub["side"].shift(), gap=sub["filled_at"].diff().dt.total_seconds())
        mirror = sub[(sub["account_id"] != sub["prev_acc"]) & (sub["instrument_symbol"] == sub["prev_sym"])
                     & (sub["side"] == sub["prev_side"]) & (sub["gap"] <= 300)]
        share = len(mirror) / len(sub)
        if share > 0.2:
            rows.append(_row("mirrored_accounts", "medium", trader,
                             f"{share:.0%} of its fills copy another of its accounts within 5 min.",
                             "The real risk is the sum of accounts; counting them separately dilutes it."))

    # 8. max_position_size exceeded (in contracts; unclear whether a micro counts as 1 or 1/10).
    mx = f.groupby("account_id")["position_after"].apply(lambda s: s.abs().max())
    lim = a.set_index("id")["max_position_size"]
    for acc in mx.index[mx > lim.reindex(mx.index)]:
        rows.append(_row("exceeds_max_position", "high", acc,
                         f"Max position {int(mx[acc])} contracts vs limit {int(lim[acc])}.",
                         "A rule breach, or the limit is measured differently; confirm with the client."))

    # 9. Positions open at the cut, and positions carried between sessions.
    for r in open_pos.itertuples():
        rows.append(_row("open_position_at_cut", "medium", f"{r.account_id} {r.instrument_symbol}",
                         f"{r.position:+d} @ {r.avg_price:.4f}, mark {r.mark_price}, unrealized ${r.unrealized_usd:,.2f}.",
                         "Today's P&L includes unrealized P&L; it can move before the close."))
    end_pos = f.groupby(["account_id", "instrument_symbol", "session_date"])["position_after"].last()
    carried = end_pos[(end_pos != 0) & (end_pos.index.get_level_values("session_date") < calendar["session_date"].max())]
    rows.append(_row("overnight_positions", "info", "-", f"{len(carried)} positions were open at the close of "
                     "a past session.", "If 0, per-session realized P&L is exact without settlement prices."))

    # 10. Partial fills and commission consistency.
    per_order = f.groupby("order_id").size()
    rows.append(_row("partially_filled_orders", "info", "-",
                     f"{(per_order > 1).sum()} orders have more than one fill ({len(f)} fills, {len(per_order)} orders).",
                     "Count activity in orders, not fills."))
    rate = (f["commission_usd"] / f["quantity"]).round(4)
    bad_rate = rate.groupby(f["instrument_symbol"]).nunique()
    rows.append(_row("commission_per_contract", "info", "-",
                     "Rate per contract per side: " + ", ".join(f"{s}=${v:.2f}" for s, v in
                                                                rate.groupby(f["instrument_symbol"]).first().items()),
                     "Consistent." if (bad_rate == 1).all() else "Inconsistent rates within a symbol."))

    # 11. Barely traded instruments.
    counts = f["instrument_symbol"].value_counts()
    for sym, n in counts[counts < 30].items():
        rows.append(_row("thinly_traded_instrument", "low", sym, f"Only {n} fills.",
                         "Any per-instrument statistic is noise; one fill weighs a lot ($1,000/point in CL)."))

    # 12. Prices far from their instrument's median in the same session.
    med = f.groupby(["instrument_symbol", "session_date"])["price"].transform("median")
    dev = (f["price"] / med - 1).abs()
    for r in f[dev > 0.02].assign(dev=dev).itertuples():
        rows.append(_row("price_outlier", "medium", f"{r.id} {r.account_id}",
                         f"{r.instrument_symbol} {r.side} {r.quantity} @ {r.price} on {r.session_date.date()}, "
                         f"{r.dev:.1%} from the session median.",
                         "Possible price or symbol error; check before attributing the loss to the trader."))

    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    return pd.DataFrame(rows).sort_values("severity", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
