import numpy as np
import yfinance as yf


def download(tickers: list[str], start: str, end: str):
    """Download OHLCV data in a single API call.

    Returns:
        close_df:  Adjusted close prices (dates x tickers), failed tickers dropped.
        volume_df: Daily volume (dates x tickers), columns aligned to close_df.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close_df = data["Close"].dropna(axis=1)
    volume_df = data["Volume"][close_df.columns]
    returns_df = close_df.pct_change().fillna(0)
    log_returns_df = np.log(close_df / close_df.shift(1))

    results = {
        "data": data,
        "close": close_df,
        "volume": volume_df,
        "returns": returns_df,
        "log_returns": log_returns_df,
    }
    return results
