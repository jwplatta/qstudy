"""Quick data coverage check: verify we have valid data through OOS_END."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import portfolio_utils as pu

START = "2015-01-01"
END = "2026-05-31"

print(f"Loading data {START} to {END} ...")
universe, benchmark, factors = pu.load_data(START, END)

print(f"\nUniverse returns shape: {universe.returns.shape}")
print(f"  First date: {universe.returns.index[0].date()}")
print(f"  Last date:  {universe.returns.index[-1].date()}")
print(f"  Tickers: {universe.returns.shape[1]}")

print(f"\nBenchmark (SPY) shape: {benchmark.returns.shape}")
print(f"  First date: {benchmark.returns.index[0].date()}")
print(f"  Last date:  {benchmark.returns.index[-1].date()}")

# Check for trailing NaN/zero coverage
last_30 = universe.returns.iloc[-30:]
coverage = last_30.notna().mean(axis=1)
print(f"\nLast 30 days ticker coverage (fraction non-NaN):")
print(coverage.tail(10).to_string())

print("\nDone.")
