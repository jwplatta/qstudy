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

## Notes

- Sector map is sourced from `qs.sector_map(SP500)` — verify sector assignments periodically as index composition changes.
- The beta neutralization loop is O(dates × passes) — slow for large universes. Consider vectorizing if experimenting with top-250+ universes.
- 10-day rebalance was chosen to limit turnover; test sensitivity with every=5 and every=20.
