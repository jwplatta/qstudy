# Short-Horizon Sector-Relative Momentum — Experiment Log

## Strategy Overview

A market-neutral, short-horizon momentum strategy that trades sector-relative price momentum within the S&P 500.

**Core idea:** Stocks that outperform their sector peers over a short horizon (20 days) tend to continue outperforming briefly. By going long sector leaders and short sector laggards, and then neutralizing both sector and market beta exposure, the strategy isolates a pure intra-sector momentum signal.

**Universe:** S&P 500 (~500 stocks), filtered to top 150 by liquidity  
**Benchmark:** SPY  
**Date range:** 2015-01-01 to 2023-12-31

---

## Experiments

### v0 — Baseline
**Hypothesis:** A 20-day sector-relative momentum signal, after neutralizing sector and market beta, generates positive risk-adjusted returns with low market correlation.

**Setup:**
- Signal: 20-day cumulative return minus sector average (sector-relative momentum), no skip period
- Filters: min price $5, min ADV $20M, top-150 by rolling dollar volume
- Positions: 20 long / 20 short, equal-weighted
- Risk: sector + beta neutralization (60-day rolling window, 2 passes)
- Rebalance: every 10 days

**Expected behavior:** Moderate Sharpe (0.5–1.0), low benchmark correlation (<0.2), modest drawdowns given neutralization.

**Results:** *(to be filled in after run)*

---

## Hypotheses for Future Versions

### v1 — Skip Period
**Hypothesis:** Adding a 3–5 day skip between signal measurement and trade entry avoids short-term reversal contamination at the tail of the lookback window.

### v2 — Shorter Window (10d) 
**Hypothesis:** A 10-day window captures faster momentum and decays before mean reversion sets in, improving Sharpe at the cost of higher turnover.

### v3 — Vol-Normalized Signal
**Hypothesis:** Dividing sector-relative returns by rolling realized vol (20-day) produces a Sharpe-like signal that ranks more consistently across varying volatility regimes.

### v4 — Residualized Signal (SPY strip)
**Hypothesis:** Residualizing returns against SPY before computing the sector-relative signal removes residual market beta from the raw return, producing cleaner sector alpha.

### v5 — Regime Scaling (VIX or correlation spike)
**Hypothesis:** Scaling down position size during high-correlation (risk-off) regimes reduces left-tail drawdowns without materially degrading returns in normal markets.

### v6 — Expand Universe / Tighten Liquidity
**Hypothesis:** Restricting to top-100 by ADV reduces capacity but improves signal quality for the short-horizon signal, which is more sensitive to execution costs.

---

---

## v1–v5 Cost Reduction Attempts

All failed to beat v0 net Sharpe. Key findings:

| Version | Change | Net Sharpe | Turnover | Verdict |
|---|---|---|---|---|
| v1 | rebalance=20 | 0.16 | 9.4% | Turnover cut but gross Sharpe collapsed (0.36). Signal decays fast, needs frequent rebalancing. |
| v2 | skip=5 | -0.37 | 16.6% | Destroyed signal. The edge lives in the most recent days. |
| v3 | smooth signal (3d) | -0.19 | 16.8% | Blunted signal too much. |
| v4 | n=10/10 | 0.09 | 16.7% | Benchmark corr jumped to 0.54 — sector neutralization breaks down with only 10 names. |
| v5 | skip+smooth+rebal20 | 0.08 | 9.2% | Combined: turnover halved but gross Sharpe still collapsed. |

**Conclusion:** v0's ~17% daily turnover is load-bearing — it's not noise, it's how the 20-day sector-relative momentum signal works. The gross Sharpe of 1.05 justifies accepting 4.3%/yr cost drag at 10 bps.

---

## v6 — Threshold-Triggered Rebalance (Dead End)

**Hypothesis:** Rebalance only when the top/bottom N book composition has changed enough (Jaccard overlap < threshold), avoiding turnover on days when the signal is stable.

**What we learned:**
- Day-over-day Jaccard overlap of the 20-long / 20-short book averages only **0.64 / 0.60**. The 20-day momentum signal rotates ~40% of the book every single day — this is intrinsic to the signal, not noise.
- A threshold of 0.7 fires on 91% of days. You'd need ~0.3 to skip 25% of days, at which point you only trade on violent regime shifts.
- Additionally: the beta neutralizer running daily is itself the primary turnover source. Any gate placed *before* neutralization still lets the neutralizer produce fresh weights daily. Any gate *after* neutralization causes a massive catch-up trade when it finally fires (full portfolio replacement).
- The correct architecture (combined gate + neutralization in one scaler) was implemented in v6 but still fires near-daily given the signal's natural rotation frequency.

**Result:** Threshold-triggered rebalancing is the wrong tool for a signal with intrinsic daily churn. It works well for slower signals (momentum over 60+ days) where the book is stable for weeks at a time.

---

## Notes

- Sector map is sourced from `qs.get_sector_map(SP500)` — verify sector assignments periodically as index composition changes.
- The beta neutralization loop is O(dates × passes) — slow for large universes. Consider vectorizing if experimenting with top-250+ universes.
- 10-day rebalance (v0) is the right balance: limits neutralizer reshuffling while capturing most of the signal's motion. Gross Sharpe 1.05 with 10 bps costs → net 0.68 is the realistic ceiling for this signal as designed.
