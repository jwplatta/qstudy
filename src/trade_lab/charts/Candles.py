from pathlib import Path

import mplfinance as mpf
import pandas as pd


class Candles:
    """Candlestick chart for price visualization.

    This class provides methods to load and visualize candlestick data
    from CSV files.
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
        style="charles",
        ax=None,
        show_volume=False,
    ):
        """Plot candlestick chart using mplfinance.

        Args:
            symbol: Trading symbol for labels
            interval: Candle interval in minutes (default: 5)
            start_time: Session start time in HH:MM format (default: "08:00")
            end_time: Session end time in HH:MM format (default: "15:00")
            figsize: Figure size tuple (width, height)
            style: mplfinance style (default: "charles")
            ax: Optional matplotlib axis to plot on (for subplots)
            show_volume: If True, display volume bars on twin y-axis (default: False)

        Returns:
            Tuple of (fig, ax) if ax is None, otherwise just ax
        """
        # Prepare data for mplfinance (requires OHLCV with datetime index)
        plot_df = self.df.copy()
        plot_df.set_index("datetime", inplace=True)

        # Rename columns to match mplfinance expectations (Title case)
        plot_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        # Filter to time range
        session_date = plot_df.index.date[0]
        time_start = pd.Timestamp(f"{session_date} {start_time}")
        time_end = pd.Timestamp(f"{session_date} {end_time}")
        plot_df = plot_df.loc[time_start:time_end]

        title = f"{interval}-Min Candles"
        if symbol:
            title = f"{symbol} {title}"
        title += f" - {session_date}"

        if ax is not None:
            # Plot on provided axis (for subplots)
            mpf.plot(
                plot_df,
                type="candle",
                volume=show_volume,
                style=style,
                ax=ax,
                ylabel="Price",
            )
            ax.set_title(title)
            return ax
        else:
            # Create new figure
            fig, axes = mpf.plot(
                plot_df,
                type="candle",
                volume=show_volume,
                style=style,
                figsize=figsize,
                title=title,
                ylabel="Price",
                returnfig=True,
            )
            return fig, axes

    def _prepare_data(self):
        """Prepare and validate candle data."""
        # Convert datetime column to pandas datetime
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.df = self.df.sort_values("datetime")

        # Ensure numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
