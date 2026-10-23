# Assessment Dataset (ML)

Seed data for the ArrowFin machine learning technical assessment. Everything here is fabricated -
no real customer data is included.

## Files

| File | Rows | Notes |
| --- | --- | --- |
| `brokers.csv` | 3 | Three brokerages. |
| `traders.csv` | 48 | One row per trader. A trader may hold more than one account. |
| `accounts.csv` | 54 | Accounts have types, statuses and balances. |
| `instruments.csv` | 8 | Contract specifications. Read `point_value_usd` carefully. |
| `market_prices.csv` | 8 | Marks as of the dataset cut, for unrealized profit and loss. |
| `fills.csv` | 5146 | Executions across 12 trading sessions. Timestamps are UTC, ISO 8601. |

## Dataset cut

All data is as of **2026-08-25T14:30:00Z**. Treat that as "now". The most recent session is "today"; the earlier
sessions are history.

## Things that are true about this data

These are not tricks. They are how trading data behaves. We mention them because ignoring them
silently produces numbers that look plausible and are wrong.

- **Contract sizes differ.** `MNQ` and `NQ` track the same index and are not the same size. Nor are
  `MES`/`ES` or `MCL`/`CL`. The point value is in `instruments.csv`.
- **A futures session is not a calendar day.** CME Globex opens at 17:00 US Central and runs until
  16:00 the following afternoon. One session spans two UTC dates, and one UTC date can contain the
  end of one session and the start of the next.
- **Commissions are real money**, charged per contract, per side.
- **One order can fill in several pieces.** Fills sharing an `order_id` are one order, not several.
- **Not every position is closed** at the cut. Marks are provided.
- **Not every account is active**, not every account has a balance, and not every trader has the
  same amount of history. Some started trading very recently.

## Loading it

Up to you - a script, a database, or plain pandas. We do not grade the loader.
