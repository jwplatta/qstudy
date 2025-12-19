from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


class Candles:
    """Candlestick chart with volume overlay.

    This class provides methods to load and visualize candlestick data
    with volume from CSV files.
    """

    @classmethod
    def from_file(cls, symbol, date, interval=5, data_dir="data"):
        """Load candle data from CSV file.

        Args:
            symbol: Trading symbol (e.g., 'SPX', 'ES')
            date: Date in YYYY-MM-DD format
            interval: Candle interval in minutes (default: 5)
            data_dir: Directory containing CSV data files

        Returns:
            Candles instance
        """
        filename = f"{symbol}_{interval}_min_{date}.csv"
        filepath = Path(data_dir) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        return cls(csv_path=filepath)

    def __init__(self, csv_path=None, dataframe=None):
        """Initialize Candles chart.

        Args:
            csv_path: Path to CSV file containing candle data
            dataframe: Pandas DataFrame with candle data (alternative to csv_path)
        """
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            raise ValueError("Must provide either csv_path or dataframe")

        self._prepare_data()

    def plot(
        self,
        symbol=None,
        interval=5,
        start_time="08:00",
        end_time="15:00",
        figsize=(14, 6),
        candle_width_pct=0.8,
    ):
        """Plot candlestick chart with volume.

        Args:
            symbol: Trading symbol for labels (optional, extracted from data if not provided)
            interval: Candle interval in minutes (default: 5)
            start_time: Session start time in HH:MM format (default: "08:00")
            end_time: Session end time in HH:MM format (default: "15:00")
            figsize: Figure size tuple (width, height)
            candle_width_pct: Width of candle as percentage of interval (default: 0.8 = 80% candle, 20% gap)

        Returns:
            Tuple of (fig, (ax_price, ax_vol))
        """
        # Extract data
        times = self.df["datetime"]
        opens = self.df["open"]
        highs = self.df["high"]
        lows = self.df["low"]
        closes = self.df["close"]
        volumes = self.df["volume"]

        # Convert timestamps to matplotlib numbers
        time_nums = mdates.date2num(times)

        # Calculate actual candle width considering gap
        # The candle_width_pct now represents how much of the interval the candle occupies
        # (gap is implicitly 1.0 - candle_width_pct)
        candle_width = (interval / 1440) * candle_width_pct

        # ======================
        # PLOT
        # ======================
        fig, ax_price = plt.subplots(figsize=figsize)
        ax_vol = ax_price.twinx()

        # ----------------------
        # Candlesticks
        # ----------------------
        for t, o, h, l, c, v in zip(time_nums, opens, highs, lows, closes, volumes):
            is_up = c >= o
            color = "green" if is_up else "red"

            # Wick
            ax_price.vlines(t, l, h, color=color, linewidth=1)

            # Body
            body_low = min(o, c)
            body_height = abs(c - o)

            ax_price.bar(
                t,
                body_height if body_height > 0 else 0.01,  # doji visibility
                bottom=body_low,
                width=candle_width,
                color=color,
                alpha=0.9,
            )

            # Volume
            ax_vol.bar(t, v, width=candle_width, color="gray", alpha=0.3)

        # ======================
        # AXES FORMATTING
        # ======================
        session_date = self.df["datetime"].dt.date.iloc[0]

        # Set x-axis limits
        x_start = pd.Timestamp(f"{session_date} {start_time}")
        x_end = pd.Timestamp(f"{session_date} {end_time}")
        ax_price.set_xlim(x_start, x_end)

        # X-axis: half-hour ticks
        ax_price.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Labels
        ax_price.set_xlabel("Time")
        symbol_label = symbol if symbol else "Price"
        ax_price.set_ylabel(f"{symbol_label}")
        ax_vol.set_ylabel("Volume")

        # Grid (price axis only, faint)
        ax_price.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

        # Rotate time labels
        plt.setp(ax_price.get_xticklabels(), rotation=45, ha="right")

        # Title
        title = f"{interval}-Min Candles with Volume"
        if symbol:
            title = f"{symbol} {title}"
        title += f" - {session_date}"
        plt.title(title)

        plt.tight_layout()

        return fig, (ax_price, ax_vol)

    def _prepare_data(self):
        """Prepare and validate candle data."""
        # Convert datetime column to pandas datetime
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.df = self.df.sort_values("datetime")

        # Ensure numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
