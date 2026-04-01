"""Load and transform QuantConnect backtest order data into trade-level data."""

import pandas as pd


def load_orders(csv_path: str, start_date: str = "2022-04-01") -> pd.DataFrame:
    """
    Load and preprocess order data from QuantConnect backtest CSV.

    Args:
        csv_path: Path to the QuantConnect orders CSV file
        start_date: Filter data to only include orders after this date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns: [Time, Symbol, Price, Quantity, Value, entry]
        - Time is converted to America/New_York timezone
        - entry is a boolean indicating if order was placed at 3:55pm ET (entry time)

    Example:
        >>> orders = load_orders("backtest_orders.csv", start_date="2022-04-01")
        >>> print(orders.head())
    """
    # Load CSV
    data = pd.read_csv(csv_path, header=0)

    # Drop unnecessary columns
    data = data.drop(columns=["Type", "Tag", "Status"])

    # Convert Time from UTC to America/New_York timezone
    data["Time"] = pd.to_datetime(data["Time"], utc=True)
    data = data[data["Time"] > pd.Timestamp(start_date, tz="UTC")]
    data["Time"] = data["Time"].dt.tz_convert("America/New_York")

    # Reset index after filtering
    data = data.reset_index(drop=True)

    # Add entry flag: marks orders placed at 3:55pm ET (entry time)
    mask = (data["Time"].dt.hour == 15) & (data["Time"].dt.minute == 55)
    data["entry"] = mask

    return data


def load_trades(csv_path: str, start_date: str = "2022-04-01") -> pd.DataFrame:
    """
    Load and summarize trade data from CSV export.

    Args:
        csv_path: Path to the trades CSV.
        start_date: Ignore trades with entry timestamps before this date (UTC).

    Returns:
        DataFrame with columns [entry_time, exit_time, pnl, fees, total_pnl] that
        aggregates all legs belonging to the same trade. The total PnL is the sum
        of the P&L column minus the summed fees for the trade.
    """

    data = pd.read_csv(csv_path, header=0, skipinitialspace=True)
    if "Order Ids" in data.columns:
        data = data.drop(columns=["Order Ids"])

    for column in ("Entry Time", "Exit Time"):
        data[column] = pd.to_datetime(data[column], utc=True)

    cutoff = pd.Timestamp(start_date, tz="UTC")
    data = data[data["Entry Time"] >= cutoff].reset_index(drop=True)

    for column in ("Entry Time", "Exit Time"):
        data[column] = data[column].dt.tz_convert("America/New_York")

    grouped = (
        data.sort_values("Entry Time")
        .groupby(["Entry Time", "Exit Time"], as_index=False)
        .agg(pnl=("P&L", "sum"), fees=("Fees", "sum"))
    )

    grouped["total_pnl"] = grouped["pnl"] - grouped["fees"]
    grouped = grouped.rename(columns={"Entry Time": "entry_time", "Exit Time": "exit_time"})

    return grouped


def _extract_short_strikes(orders_df: pd.DataFrame, order_totals: pd.DataFrame) -> pd.DataFrame:
    """Extract short call/put strikes for each aggregated trade."""
    default_index = pd.Index(order_totals["trade"].unique(), name="trade")
    empty = pd.DataFrame(index=default_index, columns=["short_call_strike", "short_put_strike"])

    symbol_col = "symbol" if "symbol" in orders_df.columns else "Symbol" if "Symbol" in orders_df.columns else None

    if symbol_col is None:
        return empty

    legs = orders_df.merge(order_totals[["Time", "trade"]], on="Time", how="inner")
    if legs.empty:
        return empty

    # Use the first timestamp per trade (entry timestamp) to identify the opening condor legs.
    first_time = legs.groupby("trade")["Time"].transform("min")
    entry_rows = legs[legs["Time"] == first_time].copy()

    if entry_rows.empty:
        return empty

    option_parts = entry_rows[symbol_col].astype(str).str.extract(r"([PC])(\d{8})$")
    entry_rows["option_type"] = option_parts[0]
    entry_rows["strike"] = pd.to_numeric(option_parts[1], errors="coerce") / 1000

    # For iron condors, the short put is the higher put strike and the short call is the lower call strike.
    short_put = entry_rows[entry_rows["option_type"] == "P"].groupby("trade")["strike"].max()
    short_call = entry_rows[entry_rows["option_type"] == "C"].groupby("trade")["strike"].min()

    return pd.DataFrame({
        "short_call_strike": short_call,
        "short_put_strike": short_put,
    }).reindex(default_index)


def build_trade_totals(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform order-level data into trade-level P&L.

    Replicates the exact transformation from the baseline analysis notebook:
    1. Group orders by timestamp and sum values (credit received)
    2. Assign trade IDs using index // 2 (assumes alternating entry/exit pairs)
    3. Aggregate to trade level: exit time and total P&L per trade
    4. Add month column for temporal grouping
    5. Add short call/put strike columns when symbol and quantity are present

    Args:
        orders_df: DataFrame from load_orders() with Time and Value columns

    Returns:
        DataFrame with index=trade and columns:
        [exit_time, value, month, short_call_strike, short_put_strike]
        - exit_time: timestamp when trade was closed
        - value: total P&L for the trade (positive = profit)
        - month: period representing the month of exit
        - short_call_strike: short call strike for the trade (if available)
        - short_put_strike: short put strike for the trade (if available)

    Example:
        >>> orders = load_orders("backtest_orders.csv")
        >>> trades = build_trade_totals(orders)
        >>> print(trades.head())
    """
    # Group by Time and sum Value, multiply by -1 to get credit perspective
    order_totals = (
        orders_df.sort_values("Time").groupby(["Time"])["Value"].sum() * -1
    ).reset_index(name="value")

    # Assign trade IDs: index // 2 (assumes entry/exit pairs)
    order_totals["trade"] = order_totals.index // 2

    # Aggregate to trade level
    trade_totals = pd.DataFrame()
    trade_totals["exit_time"] = order_totals.groupby("trade")["Time"].max()
    trade_totals["value"] = order_totals.groupby("trade")["value"].sum()

    # Add month column for temporal grouping
    trade_totals["month"] = trade_totals["exit_time"].dt.to_period("M")

    strike_totals = _extract_short_strikes(orders_df, order_totals)
    trade_totals = trade_totals.join(strike_totals, how="left")

    return trade_totals
