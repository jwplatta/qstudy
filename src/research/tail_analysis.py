"""Analyze loss concentration and tail risk in backtest results."""

import pandas as pd


def compute_tail_contribution(
    trade_totals: pd.DataFrame,
    n_worst: list[int] = [1, 5, 10]
) -> pd.DataFrame:
    """
    Compute contribution of worst N trades to total P&L.

    Measures how much of the total losses are concentrated in extreme tail events.

    Args:
        trade_totals: DataFrame with 'value' column containing trade P&L
        n_worst: List of N values to analyze (e.g., worst 1, 5, 10 trades)

    Returns:
        DataFrame with columns:
            - n_trades: number of worst trades
            - sum: sum of those trades
            - contribution_pct: percentage contribution to total P&L

    Example:
        >>> tail = compute_tail_contribution(trade_totals, n_worst=[1, 5, 10])
        >>> print(tail)
    """
    values = trade_totals["value"]
    sorted_values = values.sort_values()
    total_pnl = values.sum()

    results = []
    for n in n_worst:
        worst_n_sum = sorted_values.head(n).sum()
        contribution_pct = worst_n_sum / total_pnl if total_pnl != 0 else 0.0

        results.append({
            "n_trades": n,
            "sum": worst_n_sum,
            "contribution_pct": contribution_pct,
        })

    return pd.DataFrame(results)


def compute_tail_contribution_pct(
    trade_totals: pd.DataFrame,
    pct_worst: list[float] = [0.01, 0.05]
) -> pd.DataFrame:
    """
    Compute contribution of worst X% of trades to total P&L.

    Args:
        trade_totals: DataFrame with 'value' column containing trade P&L
        pct_worst: List of percentages (e.g., [0.01, 0.05] for 1% and 5%)

    Returns:
        DataFrame with columns:
            - pct: percentage of trades
            - n_trades: number of trades in that percentage
            - sum: sum of those trades
            - contribution_pct: percentage contribution to total P&L

    Example:
        >>> tail = compute_tail_contribution_pct(trade_totals, pct_worst=[0.01, 0.05])
        >>> print(f"Worst 1% of trades: {tail.iloc[0]['contribution_pct']:.1%} of total P&L")
    """
    values = trade_totals["value"]
    sorted_values = values.sort_values()
    total_pnl = values.sum()
    total_trades = len(values)

    results = []
    for pct in pct_worst:
        n = int(total_trades * pct)
        if n == 0:
            n = 1  # At least 1 trade

        worst_pct_sum = sorted_values.head(n).sum()
        contribution_pct = worst_pct_sum / total_pnl if total_pnl != 0 else 0.0

        results.append({
            "pct": pct,
            "n_trades": n,
            "sum": worst_pct_sum,
            "contribution_pct": contribution_pct,
        })

    return pd.DataFrame(results)


def compute_drawdown_stats(trade_totals: pd.DataFrame) -> dict:
    """
    Compute drawdown statistics.

    Args:
        trade_totals: DataFrame with 'value' column containing trade P&L

    Returns:
        Dict with keys:
            - max_drawdown: maximum drawdown in dollar terms
            - max_drawdown_pct: maximum drawdown as percentage of running max
            - equity_curve: pd.Series of cumulative P&L
            - drawdown_series: pd.Series of drawdown at each trade

    Example:
        >>> dd = compute_drawdown_stats(trade_totals)
        >>> print(f"Max drawdown: ${dd['max_drawdown']:.2f}")
    """
    values = trade_totals["value"]

    # Calculate equity curve (cumulative P&L)
    equity = values.cumsum()

    # Calculate running maximum
    running_max = equity.cummax()

    # Calculate drawdown
    drawdown = equity - running_max

    # Calculate drawdown percentage
    # Avoid division by zero by replacing zeros with small value
    running_max_safe = running_max.replace(0, 0.01)
    drawdown_pct = drawdown / running_max_safe

    return {
        "max_drawdown": drawdown.min(),
        "max_drawdown_pct": drawdown_pct.min(),
        "equity_curve": equity,
        "drawdown_series": drawdown,
    }


def get_worst_trades(trade_totals: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Get the N worst trades sorted by P&L.

    Args:
        trade_totals: DataFrame with trade data
        n: Number of worst trades to return

    Returns:
        DataFrame with worst N trades, sorted from worst to best

    Example:
        >>> worst = get_worst_trades(trade_totals, n=10)
        >>> print(worst[['exit_time', 'value']])
    """
    return trade_totals.nsmallest(n, "value")


def get_best_trades(trade_totals: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Get the N best trades sorted by P&L.

    Args:
        trade_totals: DataFrame with trade data
        n: Number of best trades to return

    Returns:
        DataFrame with best N trades, sorted from best to worst

    Example:
        >>> best = get_best_trades(trade_totals, n=10)
        >>> print(best[['exit_time', 'value']])
    """
    return trade_totals.nlargest(n, "value")


def remove_worst_trades(trade_totals: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Remove the N worst trades from the dataset.

    Useful for sensitivity analysis: "What if we could have avoided the worst losses?"

    Args:
        trade_totals: DataFrame with trade data
        n: Number of worst trades to remove

    Returns:
        DataFrame with worst N trades removed

    Example:
        >>> filtered = remove_worst_trades(trade_totals, n=5)
        >>> print(f"Removed {n} worst trades, {len(filtered)} trades remaining")
    """
    worst_trades = trade_totals.nsmallest(n, "value")
    return trade_totals.drop(worst_trades.index)
