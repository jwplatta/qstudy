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


def build_trade_totals(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform order-level data into trade-level P&L.

    Replicates the exact transformation from the baseline analysis notebook:
    1. Group orders by timestamp and sum values (credit received)
    2. Assign trade IDs using index // 2 (assumes alternating entry/exit pairs)
    3. Aggregate to trade level: exit time and total P&L per trade
    4. Add month column for temporal grouping

    Args:
        orders_df: DataFrame from load_orders() with Time and Value columns

    Returns:
        DataFrame with index=trade and columns: [exit_time, value, month]
        - exit_time: timestamp when trade was closed
        - value: total P&L for the trade (positive = profit)
        - month: period representing the month of exit

    Example:
        >>> orders = load_orders("backtest_orders.csv")
        >>> trades = build_trade_totals(orders)
        >>> print(trades.head())
    """
    # Group by Time and sum Value, multiply by -1 to get credit perspective
    order_totals = (
        orders_df.sort_values("Time")
        .groupby(["Time"])["Value"]
        .sum()
        * -1
    ).reset_index(name="value")

    # Assign trade IDs: index // 2 (assumes entry/exit pairs)
    order_totals["trade"] = order_totals.index // 2

    # Aggregate to trade level
    trade_totals = pd.DataFrame()
    trade_totals["exit_time"] = order_totals.groupby("trade")["Time"].max()
    trade_totals["value"] = order_totals.groupby("trade")["value"].sum()

    # Add month column for temporal grouping
    trade_totals["month"] = trade_totals["exit_time"].dt.to_period("M")

    return trade_totals
