#!/usr/bin/env python3
"""
Calculate a simple 9-trading-day IV-RV spread using VIX9D and SPX daily closes.

This script uses a pandas pipeline instead of row-by-row iteration:

1. Use VIX9D close on entry day t as the implied-vol proxy.
2. Use SPX close on day t as the starting price.
3. Reference the SPX close 9 trading days forward.
4. Compute forward 9-day realized volatility from the 9 daily log returns.
5. Compute:

   IV_pct = VIX9D_t
   RV_pct = 100 * sqrt((252 / 9) * sum(r_i^2))
   Spread_pct = IV_pct - RV_pct

6. Also compute the simpler move-based measure:

   ExpectedMovePct = VIX9D_t * sqrt(9 / 252)
   ActualMovePct = 100 * abs(ln(S_t+9 / S_t))
   MoveSpreadPct = ExpectedMovePct - ActualMovePct

This is a coarse diagnostic for short-vol conditions. It is not a precise
options-pricing measure for a specific SPXW chain.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

VIX9D_PATH = Path("/Users/jplatta/.schwab_rb/data/history/VIX9D_day.csv")
SPX_PATH = Path("/Users/jplatta/.schwab_rb/data/history/SPX_day.csv")
EVENTS_PATH = Path("data/research/macro_event_dates_2020_2026-04-06.csv")
OUTPUT_PATH = Path("tmp/vix9d_spx_9d_iv_rv_spreads.csv")

FORWARD_DAYS = 9
TRADING_DAYS_PER_YEAR = 252
START_DATE = "2022-01-01"
END_DATE = "2026-04-06"
ROLLING_WINDOW_DAYS = 3
SHORT_STRIKE_STD_DEVS = 1.5

OUTPUT_COLUMNS = [
    "entry_date",
    "end_date",
    "spx_start",
    "spx_end",
    "vix9d_close",
    "iv_pct",
    "trailing_avg_rv_pct",
    "rv_pct",
    "spread_pct",
    "expected_move_pct",
    "expected_move_pts",
    "actual_move_pct",
    "move_spread_pct",
    "short_put_strike",
    "short_call_strike",
    "short_put_breached",
    "short_call_breached",
    "either_short_breached",
    "synthetic_condor_outcome",
    "events",
    "status",
]


def load_daily_closes(path: Path) -> pd.DataFrame:
    """
    Load daily close values as a date-indexed DataFrame.
    """
    frame = pd.read_csv(path, usecols=["datetime", "close"])
    frame["date"] = frame["datetime"].str[:10]
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[["date", "close"]].drop_duplicates(subset=["date"], keep="last")
    return frame.sort_values("date").reset_index(drop=True)


def load_events(path: Path) -> pd.DataFrame:
    """
    Load event dates with one semicolon-joined event string per day.
    """
    if not path.exists():
        return pd.DataFrame(columns=["date", "events"])

    frame = pd.read_csv(path, usecols=["date", "event"])
    frame = frame[(frame["date"] >= START_DATE) & (frame["date"] <= END_DATE)]
    if frame.empty:
        return pd.DataFrame(columns=["date", "events"])

    events = (
        frame.groupby("date", sort=False)["event"]
        .agg(lambda values: ";".join(str(value) for value in values))
        .reset_index(name="events")
    )
    return events


def build_trailing_rv_average(spx_closes: pd.DataFrame) -> pd.DataFrame:
    """
    Build the trailing average of backward-looking 9-day annualized realized vol.
    """
    frame = spx_closes.copy()
    frame["log_return"] = (frame["close"] / frame["close"].shift(1)).map(math.log)
    frame["squared_return"] = frame["log_return"] ** 2

    frame["historical_rv_decimal"] = (
        (TRADING_DAYS_PER_YEAR / FORWARD_DAYS)
        * frame["squared_return"].rolling(window=FORWARD_DAYS, min_periods=FORWARD_DAYS).sum()
    ) ** 0.5

    frame["trailing_avg_rv_decimal"] = (
        frame["historical_rv_decimal"]
        .rolling(
            window=ROLLING_WINDOW_DAYS,
            min_periods=ROLLING_WINDOW_DAYS,
        )
        .mean()
    )

    return frame[["date", "trailing_avg_rv_decimal"]]


def build_forward_rv_inputs(spx_closes: pd.DataFrame) -> pd.DataFrame:
    """
    Build forward-looking SPX fields needed for IV-RV and condor diagnostics.
    """
    frame = spx_closes.copy()
    frame["log_return"] = (frame["close"] / frame["close"].shift(1)).map(math.log)
    frame["squared_return"] = frame["log_return"] ** 2

    future_squared_sum = sum(
        frame["squared_return"].shift(-offset) for offset in range(1, FORWARD_DAYS + 1)
    )

    frame["end_date"] = frame["date"].shift(-FORWARD_DAYS)
    frame["spx_end"] = frame["close"].shift(-FORWARD_DAYS)
    frame["rv_decimal"] = ((TRADING_DAYS_PER_YEAR / FORWARD_DAYS) * future_squared_sum) ** 0.5

    return frame[["date", "close", "end_date", "spx_end", "rv_decimal"]].rename(
        columns={"close": "spx_start"}
    )


def build_output_frame() -> pd.DataFrame:
    """
    Build the full output dataset as a pandas DataFrame.
    """
    vix9d = load_daily_closes(VIX9D_PATH).rename(columns={"close": "vix9d_close"})
    spx = load_daily_closes(SPX_PATH)
    events = load_events(EVENTS_PATH)
    trailing_rv = build_trailing_rv_average(spx)
    forward_inputs = build_forward_rv_inputs(spx)

    dataset = (
        vix9d.merge(forward_inputs, on="date", how="inner")
        .merge(trailing_rv, on="date", how="left")
        .merge(events, on="date", how="left")
        .rename(columns={"date": "entry_date"})
    )

    dataset = dataset[
        (dataset["entry_date"] >= START_DATE) & (dataset["entry_date"] <= END_DATE)
    ].copy()

    dataset["events"] = dataset["events"].fillna("")

    ok_mask = dataset["end_date"].notna() & dataset["spx_end"].notna()

    dataset["iv_pct"] = dataset["vix9d_close"]
    dataset["rv_pct"] = dataset["rv_decimal"] * 100.0
    dataset["spread_pct"] = dataset["iv_pct"] - dataset["rv_pct"]

    iv_decimal = dataset["iv_pct"] / 100.0
    expected_move_decimal = iv_decimal * math.sqrt(FORWARD_DAYS / TRADING_DAYS_PER_YEAR)
    dataset["expected_move_pct"] = expected_move_decimal * 100.0
    dataset["expected_move_pts"] = dataset["spx_start"] * expected_move_decimal
    dataset["actual_move_pct"] = (
        100.0 * (dataset["spx_end"] / dataset["spx_start"]).map(math.log).abs()
    )
    dataset["move_spread_pct"] = dataset["expected_move_pct"] - dataset["actual_move_pct"]

    width_points = SHORT_STRIKE_STD_DEVS * dataset["expected_move_pts"]
    dataset["short_put_strike"] = dataset["spx_start"] - width_points
    dataset["short_call_strike"] = dataset["spx_start"] + width_points
    dataset["short_put_breached"] = dataset["spx_end"] < dataset["short_put_strike"]
    dataset["short_call_breached"] = dataset["spx_end"] > dataset["short_call_strike"]
    dataset["either_short_breached"] = (
        dataset["short_put_breached"] | dataset["short_call_breached"]
    )
    dataset["short_put_breached"] = dataset["short_put_breached"].astype("boolean")
    dataset["short_call_breached"] = dataset["short_call_breached"].astype("boolean")
    dataset["either_short_breached"] = dataset["either_short_breached"].astype("boolean")
    dataset["synthetic_condor_outcome"] = dataset["either_short_breached"].map(
        lambda breached: "loss" if breached else "win"
    )

    dataset["trailing_avg_rv_pct"] = dataset["trailing_avg_rv_decimal"] * 100.0
    dataset["status"] = "ok"
    dataset.loc[~ok_mask, "status"] = "missing_forward_spx_window"

    missing_fields = [
        "end_date",
        "spx_end",
        "rv_pct",
        "spread_pct",
        "expected_move_pct",
        "expected_move_pts",
        "actual_move_pct",
        "move_spread_pct",
        "short_put_strike",
        "short_call_strike",
        "short_put_breached",
        "short_call_breached",
        "either_short_breached",
        "synthetic_condor_outcome",
    ]
    dataset.loc[~ok_mask, missing_fields] = pd.NA

    return dataset[OUTPUT_COLUMNS].sort_values("entry_date").reset_index(drop=True)


def format_output_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Format numeric and boolean columns to match the existing CSV output style.
    """
    formatted = dataset.copy()

    two_decimals = [
        "spx_start",
        "spx_end",
        "expected_move_pts",
        "short_put_strike",
        "short_call_strike",
    ]
    four_decimals = [
        "vix9d_close",
        "iv_pct",
        "trailing_avg_rv_pct",
        "rv_pct",
        "spread_pct",
        "expected_move_pct",
        "actual_move_pct",
        "move_spread_pct",
    ]
    boolean_columns = ["short_put_breached", "short_call_breached", "either_short_breached"]

    for column in two_decimals:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )

    for column in four_decimals:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )

    for column in boolean_columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else str(bool(value)).lower()
        )

    formatted["end_date"] = formatted["end_date"].fillna("")
    formatted["synthetic_condor_outcome"] = formatted["synthetic_condor_outcome"].fillna("")

    return formatted


def main() -> None:
    """
    Main script entrypoint.

    Usage:
        python bin/calculate_vix9d_spx_9d_iv_rv_spreads.py

    Output:
        tmp/vix9d_spx_9d_iv_rv_spreads.csv
    """
    if not VIX9D_PATH.exists():
        print(f"Missing VIX9D file: {VIX9D_PATH}")
        return

    if not SPX_PATH.exists():
        print(f"Missing SPX file: {SPX_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = format_output_frame(build_output_frame())
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(output)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
