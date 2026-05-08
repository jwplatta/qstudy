import pandas as pd


def run(positions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Core backtest engine. Applies a 1-day execution lag.

    pnl = positions.shift(1) * returns
    port_ret = pnl.sum(axis=1)

    Args:
        positions: Dollar-neutral weight DataFrame (dates x tickers).
                   Typically the output of portfolio.rebalance().
        returns:   Daily returns (dates x tickers), aligned to positions.

    Returns:
        port_ret: Daily portfolio return Series (dates,).
    """
    pnl = positions.shift(1) * returns
    return pnl.sum(axis=1)
