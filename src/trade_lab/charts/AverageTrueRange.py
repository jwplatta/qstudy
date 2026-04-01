"""Average True Range (ATR) chart for intraday range analysis."""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


class AverageTrueRange:
    """
    Chart class for visualizing Average True Range (ATR) over time.

    ATR measures the expected bar movement and helps distinguish balance vs pause:
    - Expanding ATR: impulse / pause risk
    - Stable or declining ATR: balance potential

    Balance width is typically 2-3x the 5-min ATR.
    """

    def __init__(
        self,
        symbol="ES",
        date=None,
        interval=5,
        atr_period=14,
        data_dir="data",
        debug=False,
    ):
        """
        Initialize AverageTrueRange chart.

        Args:
            symbol: The ticker symbol (e.g., 'ES' for ES futures)
            date: Date as string 'YYYY-MM-DD' (defaults to today)
            interval: Bar interval in minutes (default: 5)
            atr_period: Number of bars for ATR calculation (default: 14)
            data_dir: Directory containing OHLC CSV files
            debug: Verbose output flag
        """
        self.symbol = symbol
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.interval = interval
        self.atr_period = atr_period
        self.data_dir = Path(data_dir)
        self.debug = debug

        self.df = None
        self.atr = None

    def load_data(self):
        """
        Load OHLC data from CSV file.

        Expected filename pattern: {symbol}_{interval}_min_{date}.csv
        Expected columns: time/datetime, open, high, low, close
        """
        pattern = f"{self.symbol}_{self.interval}_min_{self.date}.csv"

        if self.debug:
            print(f"Searching for OHLC CSV file with pattern: {pattern}")

        csv_files = list(self.data_dir.glob(pattern))

        if not csv_files:
            raise ValueError(
                f"No OHLC CSV file found for {self.symbol} on {self.date} "
                f"with pattern {pattern} in {self.data_dir}"
            )

        csv_file = csv_files[0]

        if self.debug:
            print(f"Loading OHLC data from {csv_file}")

        self.df = pd.read_csv(csv_file)

        # Parse time column
        time_col = None
        for col in ["time", "datetime", "timestamp", "date"]:
            if col in self.df.columns:
                time_col = col
                break

        if time_col is None:
            raise ValueError("No time column found in OHLC data")

        self.df["datetime"] = pd.to_datetime(self.df[time_col])
        self.df = self.df.set_index("datetime").sort_index()

        # Ensure numeric columns
        for col in ["open", "high", "low", "close"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self._calculate_atr()

    def _calculate_atr(self):
        """Calculate True Range and Average True Range."""
        df = self.df

        # True Range = max(H-L, |H-Prev Close|, |L-Prev Close|)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # ATR = rolling mean of True Range
        self.atr = tr.rolling(self.atr_period).mean()
        self.df["atr"] = self.atr

        if self.debug:
            print(f"Calculated ATR with period {self.atr_period}")
            print(f"ATR range: {self.atr.min():.2f} - {self.atr.max():.2f}")

    def plot(self, figsize=(12, 6), start_time=None, end_time=None, save_path=None):
        """
        Generate and display the ATR plot.

        Args:
            figsize: Figure size (width, height) in inches
            start_time: Optional start time filter (HH:MM or HH:MM:SS)
            end_time: Optional end time filter (HH:MM or HH:MM:SS)
            save_path: Optional path to save the figure

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if self.df is None:
            self.load_data()

        plot_df = self.df.copy()

        # Apply time filters
        if start_time:
            start_dt = pd.to_datetime(f"{self.date} {start_time}")
            plot_df = plot_df[plot_df.index >= start_dt]

        if end_time:
            end_dt = pd.to_datetime(f"{self.date} {end_time}")
            plot_df = plot_df[plot_df.index <= end_dt]

        atr_series = plot_df["atr"].dropna()

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(atr_series.index, atr_series.values, linewidth=2, color="steelblue")

        # Add reference lines for interpretation
        atr_mean = atr_series.mean()
        ax.axhline(
            atr_mean,
            linestyle="--",
            color="gray",
            alpha=0.7,
            label=f"Mean ATR = {atr_mean:.2f}",
        )

        # Format x-axis for intraday with 30-minute ticks starting at 8:30
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.set_xlim(left=pd.to_datetime(f"{self.date} 08:30"))
        plt.xticks(rotation=45)

        ax.set_xlabel("Time")
        ax.set_ylabel("ATR (Points)")
        ax.set_title(f"{self.symbol} {self.interval}-Minute ATR ({self.date})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            if save_path is True:
                save_path = f"{self.symbol}_atr_{self.date}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
