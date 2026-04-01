from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class CompareVIX:
    def __init__(self, date=None, interval=10, data_dir="data"):
        """
        Initialize CompareVIX chart.

        Args:
            date: Date string in format 'YYYY-MM-DD'. Defaults to today.
            interval: Time interval in minutes. Defaults to 10.
            data_dir: Directory containing candle CSV files. Defaults to 'data'.
        """
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.interval = interval
        self.data_dir = Path(data_dir)
        self.symbols = ["VIX", "VIX1D", "VIX9D"]
        self.data = {}

    def plot(self, figsize=(14, 6)):
        """Plot close prices for VIX, VIX1D, and VIX9D.

        Args:
            figsize: Figure size tuple (width, height). Defaults to (14, 6).

        Returns:
            Tuple of (fig, ax)
        """
        if not self.data:
            self.load_data()

        if not self.data:
            print("No data available to plot")
            return

        fig, ax = plt.subplots(figsize=figsize)

        for symbol in self.symbols:
            if symbol in self.data:
                df = self.data[symbol]
                ax.plot(df["datetime"], df["close"], label=symbol, linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Close Price")
        ax.set_title(f"VIX Comparison - {self.date} ({self.interval} min intervals)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis to show time nicely
        import matplotlib.dates as mdates

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        return fig, ax

    def load_data(self):
        """Load data for VIX, VIX1D, and VIX9D from CSV files."""
        for symbol in self.symbols:
            file_path = self.data_dir / f"{symbol}_{self.interval}_min_{self.date}.csv"

            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping {symbol}")
                continue

            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            self.data[symbol] = df
