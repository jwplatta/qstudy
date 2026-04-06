#!/usr/bin/env python3
"""
Calculate a simple 7DTE-ish IV-RV spread using VIX9D and SPX daily closes.

This script follows the approximation described in the notes:

1. Use VIX9D close on entry day t as the implied-vol proxy.
2. Use SPX close on day t as the starting price.
3. Collect the next 5 SPX trading-day closes.
4. Compute forward 5-day realized volatility from the 5 daily log returns.
5. Compute:

   IV_pct = VIX9D_t
   RV_pct = 100 * sqrt((252 / 5) * sum(r_i^2))
   Spread_pct = IV_pct - RV_pct

6. Also compute the simpler move-based measure:

   ExpectedMovePct = VIX9D_t * sqrt(5 / 252)
   ActualMovePct = 100 * abs(ln(S_t+5 / S_t))
   MoveSpreadPct = ExpectedMovePct - ActualMovePct

This is a coarse diagnostic for 7DTE short-vol conditions. It is not a
precise options-pricing measure for a specific SPXW chain.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


VIX9D_PATH = Path("/Users/jplatta/.schwab_rb/data/VIX9D_day_2000-01-01_2026-03-24.csv")
SPX_PATH = Path("/Users/jplatta/.schwab_rb/data/SPX_day_2000-01-01_2026-03-24.csv")
EVENTS_PATH = Path("data/research/macro_event_dates_2020_2026-04-06.csv")
OUTPUT_PATH = Path("tmp/vix9d_spx_5d_iv_rv_spreads.csv")
FORWARD_DAYS = 5
TRADING_DAYS_PER_YEAR = 252
START_DATE = "2020-01-01"
END_DATE = "2026-04-06"


def load_daily_closes(path: Path) -> dict[str, float]:
    """
    Load daily close values keyed by YYYY-MM-DD.
    """
    closes: dict[str, float] = {}

    with path.open() as handle:
        for row in csv.DictReader(handle):
            closes[row["datetime"][:10]] = float(row["close"])

    return closes


def load_events(path: Path) -> dict[str, list[str]]:
    """
    Load event dates keyed by YYYY-MM-DD.

    Multiple events can occur on the same day, so values are stored as lists.
    """
    events_by_day: dict[str, list[str]] = {}

    if not path.exists():
        return events_by_day

    with path.open() as handle:
        for row in csv.DictReader(handle):
            day = row["date"]
            if day < START_DATE or day > END_DATE:
                continue
            events_by_day.setdefault(day, []).append(row["event"])

    return events_by_day


def main() -> None:
    """
    Main script entrypoint.

    Usage:
        python bin/calculate_vix9d_spx_5d_iv_rv_spreads.py

    Output:
        tmp/vix9d_spx_5d_iv_rv_spreads.csv
    """
    if not VIX9D_PATH.exists():
        print(f"Missing VIX9D file: {VIX9D_PATH}")
        return

    if not SPX_PATH.exists():
        print(f"Missing SPX file: {SPX_PATH}")
        return

    vix9d_closes = load_daily_closes(VIX9D_PATH)
    spx_closes = load_daily_closes(SPX_PATH)
    events_by_day = load_events(EVENTS_PATH)
    spx_days = sorted(spx_closes)
    vix9d_days = sorted(vix9d_closes)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "entry_date",
                "end_date",
                "spx_start",
                "spx_end",
                "vix9d_close",
                "iv_pct",
                "rv_pct",
                "spread_pct",
                "expected_move_pct",
                "actual_move_pct",
                "move_spread_pct",
                "events",
                "status",
            ]
        )

        for entry_date in vix9d_days:
            if entry_date < START_DATE or entry_date > END_DATE:
                continue
            if entry_date not in spx_closes:
                continue

            events_text = ";".join(events_by_day.get(entry_date, []))

            entry_index = spx_days.index(entry_date)

            # We need 5 forward trading days after the entry day:
            # t+1, t+2, t+3, t+4, t+5
            if entry_index + FORWARD_DAYS >= len(spx_days):
                writer.writerow(
                    [
                        entry_date,
                        "",
                        f"{spx_closes[entry_date]:.2f}",
                        "",
                        f"{vix9d_closes[entry_date]:.4f}",
                        f"{vix9d_closes[entry_date]:.4f}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        events_text,
                        "missing_forward_spx_window",
                    ]
                )
                row_count += 1
                continue

            spx_window_days = spx_days[entry_index : entry_index + FORWARD_DAYS + 1]
            if spx_window_days[0] != entry_date:
                continue

            spx_window_closes = [spx_closes[day] for day in spx_window_days]

            # The realized-vol formula uses the 5 forward daily log returns.
            squared_returns = []
            for previous_close, current_close in zip(spx_window_closes[:-1], spx_window_closes[1:]):
                log_return = math.log(current_close / previous_close)
                squared_returns.append(log_return * log_return)

            rv_decimal = math.sqrt((TRADING_DAYS_PER_YEAR / FORWARD_DAYS) * sum(squared_returns))
            rv_pct = rv_decimal * 100.0

            iv_pct = vix9d_closes[entry_date]
            spread_pct = iv_pct - rv_pct

            spx_start = spx_window_closes[0]
            spx_end = spx_window_closes[-1]

            expected_move_pct = iv_pct * math.sqrt(FORWARD_DAYS / TRADING_DAYS_PER_YEAR)
            actual_move_pct = 100.0 * abs(math.log(spx_end / spx_start))
            move_spread_pct = expected_move_pct - actual_move_pct

            writer.writerow(
                [
                    entry_date,
                    spx_window_days[-1],
                    f"{spx_start:.2f}",
                    f"{spx_end:.2f}",
                    f"{vix9d_closes[entry_date]:.4f}",
                    f"{iv_pct:.4f}",
                    f"{rv_pct:.4f}",
                    f"{spread_pct:.4f}",
                    f"{expected_move_pct:.4f}",
                    f"{actual_move_pct:.4f}",
                    f"{move_spread_pct:.4f}",
                    events_text,
                    "ok",
                ]
            )
            row_count += 1

    print(f"Saved {row_count} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
