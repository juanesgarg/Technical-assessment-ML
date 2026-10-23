# ArrowFin ML Engineer Assessment

## Time spent and LLM use

**Time spent:** about 2 hours.

**LLM use:** I used Claude as a tutor and pair-programmer. It proposed the code step by step (loading, CME session mapping, average-cost P&L, data checks, trader baseline and ranking), and it drafted the code review, the client reply and the slides. I ran every step, checked the outputs against an independent cash-flow check, and made the definitional choices. [TODO].

## Background

**Self-ratings** (1 never used, 2 tutorial-level, 3 shipped work with it, 4 comfortable in production, 5 could teach it):

- Python: 4
- pandas: 4
- SQL: 4
- Feature engineering: 4
- Model evaluation and validation: 4
- Time-series or sequential data: 4
- Taking a model into production: 3
- Experiment tracking: 3

**Five short answers:**

1. *What kind of modelling problem are you strongest with?* Time-series problems built on noisy sensor data, like predicting equipment failures from current, phase imbalance and runtime signals. I'm strongest at feature engineering from physical signals, because I built the data pipeline from the sensor up and know where the noise comes from.
2. *What would be the steepest learning curve for you here?* Moving from rule-based systems to rigorously evaluated predictive models. My production alerts use thresholds, so I'm still building experience with labelling, temporal validation and measuring model quality at scale. [TODO: what I'm doing to close that gap]
3. *What is something you consider a core strength that we haven't asked about?* Root-cause debugging. I trace problems to the actual cause with evidence instead of patching symptoms. That matters in risk work, where a model that looks right can be quietly wrong.
4. *What kind of work do you most want to do day-to-day?* Building new things: prototyping, testing hypotheses with data, and taking what works to production, owning it from start to finish.
5. *What do you most want to avoid?* Purely repetitive work with no room to improve it. I'm fine with monitoring and maintenance when I own what I built, but not with routine that has no learning or improvement in it.

## How to run

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then open each notebook and run Restart & Run All.

- `notebooks/01_exploration.ipynb` is the step-by-step exploratory analysis (data checks, CME sessions, P&L, the T-020 drill-down and the slide chart).
- `notebooks/02_ranking.ipynb` reproduces the same results calling the functions in `src/arrowfin/`; it is the one that produces the ranked ten. Its first cell adds `src/` to `sys.path`.

Both give the same numbers. I tested them on pandas 2.2 and 3.0 and got the same top 10 and the same net total (−67,352). Outputs are written to `outputs/` (`top10.csv`, `no_history.csv`, `data_quality.csv`, `account_sessions.csv`).

## Task 1: Traders in trouble

### How I built the numbers

1. **Load and validate with `assert`:** unique ids, foreign keys, valid sides and prices, and each order belongs to one account, one symbol and one side. My rule: broken data stops the code; strange but valid data gets reported.
2. **Timestamps in UTC** (`utc=True`). The cut is `2026-08-25T14:30Z`, which is 09:30 CT. There are no fills after the cut.
3. **CME session, not calendar day:** Chicago time + 7 h, and that date is the trade date. I use the `America/Chicago` timezone so daylight saving is handled. I checked that there are no fills between 16:00 and 17:00 CT and no weekend sessions. There are 12 sessions (Aug 10 to Aug 25). 10 fills on Aug 24 between 22:00 and 23:06 UTC belong to the Aug 25 session (all from T-020 and T-021).
4. **Orders, not fills:** 5,146 fills are 4,886 orders (260 orders were partially filled).
5. **Point value** from `instruments.csv`: ES 50, MES 5, NQ 20, MNQ 2, CL 1000, MCL 100, GC 100, 6E 125000.
6. **Realized P&L with average cost**, walking the fills in order (`filled_at`, then `id` to break ties on equal timestamps). I cross-checked it against cash flow on flat positions: the difference is 1e-10.
7. **Commissions** are subtracted per fill (a fixed rate per contract per side for each symbol, consistent across the data). **Unrealized:** 8 positions are open, all from today, valued at the mark. There are 0 overnight positions in historical sessions, so historical P&L is exact without settlement prices. I don't deduct the pending exit commission.
8. **Totals:** gross −43,209 · commissions 22,099 · unrealized −2,044 · **net −67,352**. Commissions are half the gross loss.

### What "unusually bad" means

- **Trader = person.** I add up each trader's accounts per session (`merge` with `validate="many_to_one"`) and exclude the demo account. The firm absorbs the loss across all of a trader's accounts, and if you look at them separately, T-018 (3 mirrored accounts) gets diluted.
- **Baseline per trader, using ONLY prior sessions** (no look-ahead): median net P&L, MAD × 1.4826 (a robust spread), median contracts, and worst prior session. Minimum 5 prior sessions.
- **Score:** `z = (today's P&L − median) / max(MAD, $50)`. I rank **today's session (Aug 25)**, losses only. The evidence for a human reviewer is the size versus the trader's normal and whether it's their worst day.

### The ranked ten, worst first

| # | Trader | Today (USD) | z | Size vs. normal | Worst day? |
|---|---|---|---|---|---|
| 1 | T-011 | −759 | −13.1 | 1.0× | Yes |
| 2 | T-018 (3 accounts) | −1,517 | −11.1 | 0.9× | Yes |
| 3 | T-019 (2 accounts) | −1,559 | −8.3 | 1.1× | Yes |
| 4 | T-009 | −670 | −7.4 | 1.2× | Yes |
| 5 | T-020 | −2,398 | −6.1 | 1.9× | Yes |
| 6 | T-012 | −724 | −5.2 | 1.3× | Yes |
| 7 | T-023 | −2,486 | −5.0 | 1.0× | Yes (includes −$2,340 unrealized in CL) |
| 8 | T-044 | −560 | −3.4 | 0.3× | No |
| 9 | T-008 | −459 | −2.7 | 0.8× | Yes |
| 10 | T-029 | −414 | −2.6 | 0.8× | No |

Example of why it's relative: T-006 lost −$3,214 and is **not** in the list (z −1.9), because that is normal for their size. T-011 lost −$759 and is #1.

**No history (outside the ranking, review by hand):** T-024 (−$1,580, first day, 38 contracts), T-025 (account with a $0 balance), T-026.

**Warning:** today's session is only half done (16.5 of 23 h), so these numbers can still change.

### Data notes

- Today's session is partial, and 10 fills have a UTC date different from their session.
- ACC-2053 (closed, T-040) traded Aug 10 to 14; ACC-2054 (restricted, T-041) is trading on Aug 24 and 25. There is no date for the status change.
- ACC-2028 (T-025) has a $0 balance and is trading today. ACC-2047 (T-043) is a demo account with no fills.
- Short history: ACC-2027 and ACC-2028 (1 session), ACC-2029 and ACC-2054 (2 sessions).
- Several accounts per trader: T-018 (3 accounts that trade in mirror: 27% of their fills replicate another account within 5 minutes), T-019, T-040, T-041, T-043.
- "Mateo Herrera" is both T-001 (BRK-ARWP) and T-017 (BRK-SMPT), so I always identify traders by `trader_id`.
- Position limit exceeded: ACC-2013 (9 vs. 5) and ACC-2025 (32 vs. 20). This could be a violation, or micros might count as 1/10. I would ask the client.
- CL has only 3 fills ($1,000 per point). 9 MCL prices are more than 2% away from their session median.
- What is **clean:** no duplicates, no nulls, no fills after the cut, consistent commissions.

## Task 2: Labels and evaluation

### (a) Labels

A *complete* session is bad if its net P&L (realized + unrealized at the close − commissions) is ≤ −3 robust spreads below the trader's median, computed only from prior sessions (minimum 5). It is also bad if it breaks a firm rule (daily loss, drawdown, maximum size). I think this is defensible because it's relative to the trader, and the rule part is tied to the firm's real loss.

Where it's ambiguous: the threshold is arbitrary; new traders; person versus account; partial sessions; and large losses that are normal for that trader.

To check that two people agree: two reviewers label ~100 random sessions with a written guide, I measure Cohen's kappa (target > 0.7), and adjust the guide.

**First model:** logistic regression as a baseline, then gradient boosting.
- **Features** (past information only): P&L so far / spread, size versus normal, orders versus normal, position versus limit, losing streak, account age and account type.
- **Target:** bad session.
- **Split:** **walk-forward by date** (train on sessions 1..k, validate on k+1), never random. I would also hold out whole traders to measure how it does on new traders.

### (b) Evaluation

I would measure precision@10 against real outcomes (liquidations or broken rules), recall of liquidations, lead time (how many hours before the loss), and how stable the list is from day to day.

What this dataset doesn't give me: real outcomes, limits per account, status change dates, intraday and settlement prices, more history, and reviewers' opinions.

**One specific risk: circularity.** If the label is defined with the same z as the ranking, the ranking will be right almost 100% of the time without proving anything. It has to be evaluated against an independent outcome. A second risk: `status` is taken at the cut (ACC-2053 is already closed), so using it as a feature for past sessions means using information from the future.

## Task 3: Slides

[TODO: slides link, added once the repo is on GitHub]

## Task 4: Code review

**Verdict: I would not approve it.**

**Blocking:**

1. **Look-ahead (the one I'd fix first):** `sessions[:i + 1]` includes the current session, so `avg_session_pnl` contains the same `net_pnl` that defines `label_bad`. Fix: `sessions[:i]`.
2. **Mixes traders:** the history accumulates sessions from all traders. Fix: group by `trader_id`.
3. **Hardcoded `POINT_VALUE = 20`:** only correct for NQ (MNQ is 2, ES 50, CL 1000, 6E 125000). Also, `qty × avg_price` doesn't work if a session mixes instruments. Fix: point value per fill from `instruments.csv`.
4. **Division by zero** if `minutes_active == 0`.

**Comments (non-blocking):**

- `label_bad < -1000` is an absolute threshold. It should be relative to the trader, and it should live outside the features function.
- Use the median instead of the mean.
- `session_date` should be the CME session, not the calendar date.
- No tests. For example: "the feature for session i doesn't change if I modify the `net_pnl` of session i".

## Task 5: Client reply

> Hi R., thanks for flagging this. T-020 is partly right: they traded only MES, as usual, and their largest position this session (6 contracts) is in line with their history. What changed is activity and timing: they placed 10 orders versus exactly 6 in each of their previous 11 sessions (52 contracts vs. about 28), and they traded the 17:00 CT evening open on Aug 24, a window they had never traded before, where about $1,800 of the loss happened. Note that a calendar-day (UTC) report would show those evening trades under Aug 24, which may be why the trader doesn't see an unusual day. As of 09:30 CT their session is at −$2,398 net of commissions, about six times their usual daily swing and worse than any of their prior sessions. The flag compares each trader only against their own history; it isn't a judgment on skill, and the session isn't closed yet, so the number can still move. I'd suggest the account manager share this breakdown with the trader rather than a verdict. I'll re-run the numbers at the session close and send you the trade-by-trade detail by end of day.

## Task 6: Reflection

**The decision I'm most proud of** is measuring each trader against their own prior history (no look-ahead) and adding up each person's accounts. The cost: I can't rank new traders (T-024 lost $1,580 on their first day and stayed out of the top 10), and with only 11 sessions the baseline is noisy.

**With a second day,** I would evaluate the ranking against real outcomes and compare today's partial session against the same time window of prior sessions. Before starting, I would ask you for: liquidation or rule-break history, limits per account (daily loss and drawdown), status change dates, how micros count toward the position limit, and intraday marks.

## What I'd do with more time

- Move more of the logic into functions with tests.
- Adjust the partial session by time window.
- A peer baseline for new traders.
