#!/usr/bin/env python3
"""
Calculate simple 1DTE IV-RV spreads for SPXW chains.

This script is intentionally simple and follows the workflow below:

1. Find the last SPXW option-chain sample from the calendar day before expiry.
2. Read the option chain CSV and estimate ATM IV from the nearest strike.
3. Use the chain's underlying_price as the entry SPX level (S_t).
4. Find the next available SPX trading-day close after the entry date (S_t+1).
5. Compute:

   r = ln(S_t+1 / S_t)
   RV = sqrt(252) * abs(r)
   spread = IV - RV

6. Also compute the easier-to-interpret move version:

   expected_move = IV / sqrt(252)
   actual_move = abs(r)
   move_spread = expected_move - actual_move

Notes:
- IV is annualized ATM IV from the option chain.
- RV is annualized forward realized volatility over 1 trading day.
- The script writes CSV output into tmp/spxw_iv_rv_spreads.csv.
- SPX closes are loaded from local repo files first. If you download newer
  Schwab price-history CSVs into /tmp/schwab_rb_data, those files are used too.
"""

from __future__ import annotations

import bisect
import csv
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


CHAIN_FILENAME_PATTERN = re.compile(
    r"^SPXW_exp(?P<expiry>\d{4}-\d{2}-\d{2})_(?P<sample_date>\d{4}-\d{2}-\d{2})_(?P<sample_time>\d{2}-\d{2}-\d{2})\.csv$"
)


@dataclass
class SelectedChain:
    expiry_date: date
    sample_timestamp: datetime
    path: Path


def parse_chain_filename(path: Path) -> tuple[date, datetime] | None:
    """
    Parse one SPXW chain filename.

    Example:
        SPXW_exp2025-12-23_2025-12-22_14-30-41.csv
    """
    match = CHAIN_FILENAME_PATTERN.match(path.name)
    if match is None:
        return None

    expiry_date = datetime.strptime(match.group("expiry"), "%Y-%m-%d").date()
    sample_timestamp = datetime.strptime(
        f"{match.group('sample_date')} {match.group('sample_time')}",
        "%Y-%m-%d %H-%M-%S",
    )
    return expiry_date, sample_timestamp


def find_last_day_before_expiry_chains(data_dir: Path) -> list[SelectedChain]:
    """
    For each expiry date, keep the last chain sample from the calendar day before expiry.
    """
    selected_by_expiry: dict[date, SelectedChain] = {}

    for csv_file in sorted(data_dir.glob("SPXW_exp*.csv")):
        parsed = parse_chain_filename(csv_file)
        if parsed is None:
            continue

        expiry_date, sample_timestamp = parsed
        if sample_timestamp.date() != expiry_date - timedelta(days=1):
            continue

        current = selected_by_expiry.get(expiry_date)
        if current is None or sample_timestamp > current.sample_timestamp:
            selected_by_expiry[expiry_date] = SelectedChain(
                expiry_date=expiry_date,
                sample_timestamp=sample_timestamp,
                path=csv_file,
            )

    return [selected_by_expiry[expiry] for expiry in sorted(selected_by_expiry)]


def estimate_atm_iv(chain_path: Path) -> tuple[float, float, float] | None:
    """
    Estimate ATM IV using the nearest strike in the chain.

    The implementation is deliberately simple:
    - Read underlying_price from the chain rows
    - Find the strike closest to spot
    - Average call and put IV at that strike when both exist

    Returns:
        (underlying_price, atm_strike, atm_iv_decimal)
    """
    with chain_path.open() as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    underlying_price = float(rows[0]["underlying_price"])
    strikes = sorted({float(row["strike"]) for row in rows})
    if not strikes:
        return None

    atm_strike = min(strikes, key=lambda strike: abs(strike - underlying_price))

    iv_values = []
    for row in rows:
        if float(row["strike"]) != atm_strike:
            continue

        volatility_text = row.get("volatility", "").strip()
        if not volatility_text:
            continue

        # The CSV stores annualized IV in percent, for example 16.995.
        iv_values.append(float(volatility_text) / 100.0)

    if not iv_values:
        return None

    atm_iv = sum(iv_values) / len(iv_values)
    return underlying_price, atm_strike, atm_iv


def load_spx_closes() -> dict[str, float]:
    """
    Load SPX closes from local files.

    Load order:
    1. ~/.schwab_rb/data/SPX_day_2000-01-01_2026-03-24.csv
    2. data/research/SPX_day_1980-01-01_2026-03-24.csv
    3. data/SPX_5_min_YYYY-MM-DD.csv files (last close of each file)
    4. /tmp/schwab_rb_data/*.csv files, if present

    Later sources overwrite earlier ones.
    """
    closes: dict[str, float] = {}

    schwab_daily = Path("/Users/jplatta/.schwab_rb/data/SPX_day_2000-01-01_2026-03-24.csv")
    if schwab_daily.exists():
        with schwab_daily.open() as handle:
            for row in csv.DictReader(handle):
                closes[row["datetime"][:10]] = float(row["close"])

    research_daily = Path("data/research/SPX_day_1980-01-01_2026-03-24.csv")
    if research_daily.exists():
        with research_daily.open() as handle:
            for row in csv.DictReader(handle):
                closes[row["datetime"][:10]] = float(row["close"])

    for intraday_file in sorted(Path("data").glob("SPX_5_min_*.csv")):
        day = intraday_file.stem.replace("SPX_5_min_", "")
        with intraday_file.open() as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            closes[day] = float(rows[-1]["close"])

    # This allows a manual Schwab fetch to extend the date range without
    # changing the script again.
    for schwab_file in sorted(Path("/tmp/schwab_rb_data").glob("*.csv")):
        with schwab_file.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        for row in rows:
            date_text = ""
            if "datetime" in row:
                date_text = row["datetime"][:10]
            elif "date" in row:
                date_text = row["date"][:10]
            if date_text:
                closes[date_text] = float(row["close"])

    return closes


def find_next_trading_day_close(entry_day: date, closes: dict[str, float]) -> tuple[str, float] | None:
    """
    Find the next available close after the entry date.

    This uses the next trading day available in the close series, which handles
    weekends and market holidays in a simple way.
    """
    trading_days = sorted(closes)
    entry_day_text = entry_day.isoformat()
    next_index = bisect.bisect_right(trading_days, entry_day_text)
    if next_index >= len(trading_days):
        return None

    next_day = trading_days[next_index]
    return next_day, closes[next_day]


def format_pct(value: float) -> str:
    """
    Format a decimal as a percent string.

    Example:
        0.16995 -> 16.9950
    """
    return f"{value * 100:.4f}"


def main() -> None:
    """
    Main script entrypoint.

    Usage:
        python bin/calculate_spxw_1dte_iv_rv_spreads.py

    Output:
        tmp/spxw_iv_rv_spreads.csv
    """
    data_dir = Path("data")
    output_path = Path("tmp/spxw_iv_rv_spreads.csv")
    if not data_dir.exists():
        print("data/ directory not found")
        return

    selected_chains = find_last_day_before_expiry_chains(data_dir)
    if not selected_chains:
        print("No day-before-expiry SPXW chain files found in data/")
        return

    spx_closes = load_spx_closes()
    if not spx_closes:
        print("No SPX close data found")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "expiry_date",
                "entry_date",
                "entry_time",
                "next_trading_day",
                "entry_spx",
                "next_close_spx",
                "atm_strike",
                "iv_pct",
                "rv_pct",
                "spread_pct",
                "expected_move_pct",
                "actual_move_pct",
                "move_spread_pct",
                "chain_file",
                "status",
            ]
        )

        for chain in selected_chains:
            atm_info = estimate_atm_iv(chain.path)
            if atm_info is None:
                writer.writerow(
                    [
                        chain.expiry_date.isoformat(),
                        chain.sample_timestamp.date().isoformat(),
                        chain.sample_timestamp.strftime("%H:%M:%S"),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        str(chain.path),
                        "missing_atm_iv",
                    ]
                )
                row_count += 1
                continue

            entry_spx, atm_strike, atm_iv = atm_info
            next_close_info = find_next_trading_day_close(chain.sample_timestamp.date(), spx_closes)
            if next_close_info is None:
                writer.writerow(
                    [
                        chain.expiry_date.isoformat(),
                        chain.sample_timestamp.date().isoformat(),
                        chain.sample_timestamp.strftime("%H:%M:%S"),
                        "",
                        f"{entry_spx:.2f}",
                        "",
                        f"{atm_strike:.2f}",
                        format_pct(atm_iv),
                        "",
                        "",
                        "",
                        "",
                        "",
                        str(chain.path),
                        "missing_next_trading_day_close",
                    ]
                )
                row_count += 1
                continue

            next_trading_day, next_close_spx = next_close_info
            log_return = math.log(next_close_spx / entry_spx)
            realized_move = abs(log_return)
            rv_annualized = math.sqrt(252.0) * realized_move
            spread_annualized = atm_iv - rv_annualized
            expected_move = atm_iv / math.sqrt(252.0)
            move_spread = expected_move - realized_move

            writer.writerow(
                [
                    chain.expiry_date.isoformat(),
                    chain.sample_timestamp.date().isoformat(),
                    chain.sample_timestamp.strftime("%H:%M:%S"),
                    next_trading_day,
                    f"{entry_spx:.2f}",
                    f"{next_close_spx:.2f}",
                    f"{atm_strike:.2f}",
                    format_pct(atm_iv),
                    format_pct(rv_annualized),
                    format_pct(spread_annualized),
                    format_pct(expected_move),
                    format_pct(realized_move),
                    format_pct(move_spread),
                    str(chain.path),
                    "ok",
                ]
            )
            row_count += 1

    print(f"Saved {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()
