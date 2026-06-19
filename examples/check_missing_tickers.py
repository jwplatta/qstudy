"""Check which current S&P 500 tickers are missing data in 2026."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import portfolio_utils as pu
import qstudy as qs

START = "2015-01-01"
END = "2026-05-31"
CHECK_FROM = "2026-01-01"

print(f"Loading universe {START} to {END} ...")
universe, _, _ = pu.load_data(START, END)

# Current SP500 ticker list
sp500_tickers = set(qs.constants.SP500)
loaded_tickers = set(universe.returns.columns)

# Tickers in SP500 constant but not loaded at all
not_loaded = sp500_tickers - loaded_tickers
print(f"\nTickers in SP500 list but not loaded at all ({len(not_loaded)}):")
for t in sorted(not_loaded):
    print(f"  {t}")

# Tickers loaded but entirely NaN in 2026
oos = universe.returns.loc[CHECK_FROM:]
all_nan_in_oos = oos.columns[oos.isna().all()].tolist()
print(f"\nTickers loaded but all-NaN from {CHECK_FROM} ({len(all_nan_in_oos)}):")
for t in sorted(all_nan_in_oos):
    print(f"  {t}")

# Summary of last available date per ticker (for loaded ones with gaps)
last_valid = oos.apply(lambda col: col.last_valid_index())
stale = last_valid[last_valid < oos.index[-30]].sort_values()
print(f"\nTickers with last valid date > 30 days before end ({len(stale)}):")
for t, d in stale.items():
    print(f"  {t}: last valid {d.date()}")

print("\nDone.")
