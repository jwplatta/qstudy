MAJOR_INDEXES = ["SPY", "QQQ", "DIA", "IWM"]

SECTOR_ETFS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]

# Maps GICS sector strings to their corresponding SPDR sector ETF tickers.
# Useful for looking up which ETF proxy corresponds to a stock's sector classification.
SECTOR_ETF_MAP: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Technology": "XLK",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

VOL_INDEXES = ["^VIX", "^VIX9D", "^VIX1D"]
