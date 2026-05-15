"""
Toy script demonstrating the build_long_short_positions NaN-universe bug.

Setup:
  - 10 tickers total, only 4 are "tradeable" (have signal)
  - 6 tickers are non-tradeable → NaN signal (e.g. below liquidity threshold)
  - Mean reversion signal: negative 1-day return (losers expected to bounce)
  - We want 2 longs and 2 shorts from the 4 tradeable tickers

Run this and observe: with the buggy cutoff, the short book is empty.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hard-coded price data
# ---------------------------------------------------------------------------
# 4 tradeable tickers with clear recent moves:
#   AAPL: big winner (expect short — mean reversion)
#   MSFT: moderate winner
#   GOOG: moderate loser
#   META: big loser (expect long — mean reversion)
# 6 non-tradeable tickers (tiny/illiquid stocks): NaN signal

dates = pd.date_range("2024-01-01", periods=6, freq="B")

# fmt: off
prices = pd.DataFrame({
    # Tradeable: strong uptrend → negative MR signal (short candidate)
    "AAPL": [100, 103, 107, 112, 118, 125],
    # Tradeable: mild uptrend
    "MSFT": [100, 101, 102, 103, 104, 105],
    # Tradeable: mild downtrend
    "GOOG": [100,  99,  98,  97,  96,  95],
    # Tradeable: strong downtrend → positive MR signal (long candidate)
    "META": [100,  96,  91,  85,  78,  70],
    # Non-tradeable (illiquid): will have NaN signal
    "PENNY1": [1.00, 1.01, 0.99, 1.02, 0.98, 1.00],
    "PENNY2": [0.50, 0.51, 0.49, 0.52, 0.48, 0.50],
    "PENNY3": [2.00, 2.01, 1.99, 2.02, 1.98, 2.00],
    "TINY1":  [3.00, 3.01, 2.99, 3.02, 2.98, 3.00],
    "TINY2":  [0.10, 0.11, 0.09, 0.12, 0.08, 0.10],
    "TINY3":  [0.25, 0.26, 0.24, 0.27, 0.23, 0.25],
}, index=dates)
# fmt: on

# ---------------------------------------------------------------------------
# Step 1: compute 1-day returns as signal base (full DataFrame, all dates)
# ---------------------------------------------------------------------------
returns_1d = prices.pct_change(periods=1)  # first row is all NaN — that's fine

print("=== Raw 1-day returns (all dates) ===")
print(returns_1d.round(4))

# ---------------------------------------------------------------------------
# Step 2: apply tradeable mask — PENNY and TINY stocks are non-tradeable
# ---------------------------------------------------------------------------
tradeable = pd.Series(
    {t: True  for t in ["AAPL", "MSFT", "GOOG", "META"]} |
    {t: False for t in ["PENNY1", "PENNY2", "PENNY3", "TINY1", "TINY2", "TINY3"]}
)

# Mean reversion signal: negate return so big losers → high signal (rank 1 = best long)
signal_df = -returns_1d
signal_df.loc[:, ~tradeable] = np.nan  # mask non-tradeable columns across all dates

print("\n=== Signal after tradeable mask (NaN = non-tradeable) ===")
print(signal_df.round(4))

# ---------------------------------------------------------------------------
# Step 3: build_long_short_positions — replicated inline for full DataFrame
# ---------------------------------------------------------------------------
n_long = 2
n_short = 2

signal_rank = signal_df.rank(axis=1, ascending=False, na_option="bottom")

print("\n=== Ranks (ascending=False, na_option='bottom') ===")
print(signal_rank.astype(int))

# --- BUGGY version ---
n_total_buggy = signal_rank.count(axis=1)  # always = 10 (no NaNs after na_option='bottom')
short_cutoff_buggy = n_total_buggy - (n_short - 1)

long_mask = signal_rank <= n_long
short_mask_buggy = signal_rank.ge(short_cutoff_buggy.values[:, None])
positions_buggy = long_mask.astype(float) - short_mask_buggy.astype(float)

print("\n=== BUGGY build ===")
print(f"n_total (signal_rank.count) — every row = {n_total_buggy.iloc[-1]}  ← counts all 10 cols")
print(f"short_cutoff on last date   = {int(short_cutoff_buggy.iloc[-1])}  ← selects ranks 9-10 (NaN tickers!)")
print("\nPositions (all dates):")
print(positions_buggy)
print(f"\nLongs per date:  {(positions_buggy > 0).sum(axis=1).tolist()}")
print(f"Shorts per date: {(positions_buggy < 0).sum(axis=1).tolist()}  ← BUG: always 0")

# --- FIXED version ---
n_tradeable_count = signal_df.count(axis=1)  # counts only non-NaN = 4 (or 0 on warmup row)
short_cutoff_fixed = n_tradeable_count - (n_short - 1)

short_mask_fixed = signal_rank.ge(short_cutoff_fixed.values[:, None]) & signal_df.notna()
positions_fixed = long_mask.astype(float) - short_mask_fixed.astype(float)
abs_sum = positions_fixed.abs().sum(axis=1).replace(0, float("nan"))
positions_fixed = positions_fixed.div(abs_sum, axis=0)

print("\n=== FIXED build ===")
print(f"n_tradeable (signal_df.count) — last date = {int(n_tradeable_count.iloc[-1])}  ← counts only 4 tradeable")
print(f"short_cutoff on last date     = {int(short_cutoff_fixed.iloc[-1])}  ← selects ranks 3-4 (GOOG and AAPL)")
print("\nPositions (all dates):")
print(positions_fixed.round(4))
print(f"\nLongs per date:  {(positions_fixed > 0).sum(axis=1).tolist()}")
print(f"Shorts per date: {(positions_fixed < 0).sum(axis=1).tolist()}")

# ---------------------------------------------------------------------------
# Step 4: sanity check on the last valid date
# ---------------------------------------------------------------------------
print("\n=== Interpretation (last date) ===")
last_pos = positions_fixed.iloc[-1]
last_signal = signal_df.iloc[-1]

longs  = last_pos[last_pos > 0].index.tolist()
shorts = last_pos[last_pos < 0].index.tolist()

print(f"Long  (expect bounce):   {longs}")
print(f"Short (expect reversal): {shorts}")
print("\nSignal value on last date vs position:")
for t in longs + shorts:
    print(f"  {t}: signal={last_signal[t]:+.4f}  →  {'LONG' if t in longs else 'SHORT'}")
