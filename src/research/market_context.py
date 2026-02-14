"""Merge backtest results with market data for contextual analysis."""

import pandas as pd


def merge_market_data(
    trade_totals: pd.DataFrame, market_csv_path: str, date_column: str = "date"
) -> pd.DataFrame:
    """
    Merge trade data with market data (e.g., VIX, SPX daily data).

    Args:
        trade_totals: DataFrame with 'exit_time' column
        market_csv_path: Path to market data CSV file
        date_column: Name of date column in market CSV (default: 'date')

    Returns:
        DataFrame with market data columns added (open, high, low, close, range)
        Note: 'volume' column is dropped if present

    Example:
        >>> trades_with_market = merge_market_data(
        ...     trade_totals,
        ...     "data/VIX_daily_2022-2025.csv"
        ... )
        >>> print(trades_with_market.head())
    """
    df = trade_totals.copy()

    # Add date column from exit_time
    df["date"] = pd.to_datetime(df["exit_time"]).dt.date

    # Load market data
    market = pd.read_csv(market_csv_path, header=0)
    market["date"] = pd.to_datetime(market[date_column]).dt.date

    # Merge on date (left join to keep all trades)
    df = df.merge(market, on="date", how="left")

    # Drop volume column if it exists
    if "volume" in df.columns:
        df = df.drop(columns=["volume"])

    df["range"] = df["high"] - df["low"]

    return df


def compute_range_bucketed_stats(
    trade_totals_with_market: pd.DataFrame, range_column: str = "range", n_buckets: int = 20
) -> pd.DataFrame:
    """
    Bucket trades by market range (e.g., VIX or SPX daily range) and compute statistics.

    Args:
        trade_totals_with_market: DataFrame with market data merged
        range_column: Name of the range column to bucket by (default: 'range')
        n_buckets: Number of quantile buckets to create (default: 20)

    Returns:
        DataFrame with range_bucket as index and columns:
            - count: number of trades
            - win_rate: percentage of profitable trades
            - avg_pnl: average P&L
            - avg_range: average range value in bucket

    Example:
        >>> bucketed = compute_range_bucketed_stats(trades_with_market, n_buckets=20)
        >>> print(bucketed)
    """
    df = trade_totals_with_market.copy()

    # Create range buckets using quantiles
    df["range_bucket"] = pd.qcut(df[range_column], q=n_buckets, duplicates="drop")

    # Compute statistics per bucket
    stats = df.groupby("range_bucket", observed=False).agg(
        {
            "value": [
                ("count", "count"),
                ("win_rate", lambda x: (x > 0).mean()),
                ("avg_pnl", "mean"),
            ],
            range_column: [("avg_range", "mean")],
        }
    )

    # Flatten multi-level column names
    stats.columns = ["_".join(col).strip("_") for col in stats.columns.values]

    return stats


def compare_win_loss_market_context(
    trade_totals_with_market: pd.DataFrame, context_columns: list[str]
) -> pd.DataFrame:
    """
    Compare market context between winning and losing trades.

    Args:
        trade_totals_with_market: DataFrame with market data merged
        context_columns: List of column names to compare (e.g., ['range', 'close', 'open'])

    Returns:
        DataFrame with columns:
            - column: name of the context column
            - win_mean: mean value for winning trades
            - loss_mean: mean value for losing trades
            - difference: win_mean - loss_mean

    Example:
        >>> comparison = compare_win_loss_market_context(
        ...     trades_with_market,
        ...     context_columns=['range', 'close']
        ... )
        >>> print(comparison)
    """
    wins = trade_totals_with_market[trade_totals_with_market["value"] > 0.0]
    losses = trade_totals_with_market[trade_totals_with_market["value"] <= 0.0]

    results = []
    for col in context_columns:
        win_mean = wins[col].mean()
        loss_mean = losses[col].mean()

        results.append(
            {
                "column": col,
                "win_mean": win_mean,
                "loss_mean": loss_mean,
                "difference": win_mean - loss_mean,
            }
        )

    return pd.DataFrame(results)
