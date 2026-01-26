from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from ..utils.gex import row_gross_gex


class GrossGEX:
    """Gross Gamma Exposure (Gross GEX) calculator and plotter.

    Computes and visualizes gross gamma exposure over time to identify
    market regime (mean reversion vs trend/breakout).

    Gross GEX is dealer-agnostic and represents the absolute gamma exposure.
    """

    def __init__(
        self,
        data_dir="data",
        strike_width=50.0,
        multiplier=100.0,
        gamma_scale=0.01,
    ):
        """Initialize Gross GEX calculator and plotter.

        Args:
            data_dir: Directory containing option chain CSV files
            strike_width: Half-width of strike band around spot (default: 50.0)
                         e.g., 50 means strikes within +/- 50 points of spot
            multiplier: Contract multiplier (default: 100.0 for SPX)
            gamma_scale: Scaling factor for gamma units (default: 0.01)
        """
        self.data_dir = Path(data_dir)
        self.strike_width = strike_width
        self.multiplier = multiplier
        self.gamma_scale = gamma_scale
        self.timestamps = []
        self.net_gex_values = []
        self.strike_counts = []
        self.top5_gex_values = []
        self.top5_strikes = None

    def plot(self, figsize=(14, 7), save_path=None):
        """Plot Gross Gamma Exposure over time as a line chart.

        Args:
            figsize: Figure size (width, height) in inches (default: (14, 7))
            save_path: Optional path to save the figure (default: None)
                       Pass True to save to gross_gex.png

        Returns:
            Tuple of (fig, ax)
        """
        if not self.timestamps:
            raise ValueError("No data to plot. Call load_and_calculate() first.")

        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot GEX on primary axis
        ax1.plot(
            self.timestamps,
            self.net_gex_values,
            "b-",
            linewidth=2,
            label="Strike Window",
        )
        ax1.scatter(self.timestamps, self.net_gex_values, c="blue", s=20, zorder=5)

        # Plot Top 5 Strikes GEX if available
        if self.top5_gex_values:
            ax1.plot(
                self.timestamps,
                self.top5_gex_values,
                "r-",
                linewidth=2,
                label="Top 5 OI",
                linestyle="--",
            )
            ax1.scatter(self.timestamps, self.top5_gex_values, c="red", s=20, zorder=5)

        # Zero line
        ax1.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # Format x-axis to show time as HH:MM
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Labels and styling
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Gross Gamma Exposure")
        title = f"Intraday Gross GEX (±{self.strike_width} strike window)"
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_path:
            if save_path is True:
                save_path = "gross_gex.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax1

    def load_and_calculate(self, symbol=None, expiration_filter=None, sample_date=None):
        """Load all option chain CSV files and calculate Gross GEX for each timestamp.

        Args:
            symbol: Trading symbol to filter files (e.g., '$SPX', 'SPXW')
            expiration_filter: Expiration date string (YYYY-MM-DD) to filter files (required)
            sample_date: Optional specific date (YYYY-MM-DD) to filter files
        """
        if symbol is None:
            raise ValueError("symbol is required and cannot be None")

        if expiration_filter is None:
            raise ValueError("expiration_filter is required and cannot be None")

        if sample_date is not None:
            pattern = f"{symbol}_exp{expiration_filter}_{sample_date}_*.csv"
        else:
            pattern = f"{symbol}_exp{expiration_filter}_*.csv"

        csv_files = sorted(self.data_dir.glob(pattern))

        if not csv_files:
            filter_msg = f" for symbol {symbol}" if symbol else ""
            filter_msg += f" and expiration {expiration_filter}"
            raise ValueError(f"No option chain CSV files found in {self.data_dir}{filter_msg}")

        # Identify top 5 strikes by open interest from first file
        if csv_files:
            first_df = pd.read_csv(csv_files[0])
            if not first_df.empty and "underlying_price" in first_df.columns:
                spot_first = float(first_df["underlying_price"].iloc[0])
                # Filter to strike window
                band_first = first_df[
                    (first_df["strike"] >= spot_first - self.strike_width)
                    & (first_df["strike"] <= spot_first + self.strike_width)
                ]
                if not band_first.empty:
                    # Get top 5 strikes by open interest
                    top5_df = band_first.nlargest(5, "open_interest")
                    self.top5_strikes = set(top5_df["strike"].values)

        for csv_file in csv_files:
            try:
                # Parse timestamp from filename: $SPX_exp2025-12-24_2025-12-18_14-30-00.csv
                # Format: {symbol}_exp{expiration_date}_{fetch_date}_{fetch_time}.csv
                parts = csv_file.stem.split("_")
                if len(parts) >= 4:
                    # Parts: ['$SPX', 'exp2025-12-24', '2025-12-18', '14-30-00']
                    fetch_date = parts[2]
                    fetch_time = parts[3]
                    timestamp = datetime.strptime(f"{fetch_date}_{fetch_time}", "%Y-%m-%d_%H-%M-%S")
                else:
                    continue

                df = pd.read_csv(csv_file)

                # Calculate Gross GEX for this snapshot
                gross_gex, strike_count = self._compute_net_gex_near_spot(df)

                # Calculate Gross GEX for top 5 strikes if identified
                top5_gex = 0.0
                if self.top5_strikes:
                    top5_gex = self._compute_top5_gex(df)

                self.timestamps.append(timestamp)
                self.net_gex_values.append(gross_gex)
                self.strike_counts.append(strike_count)
                self.top5_gex_values.append(top5_gex)

            except Exception as e:
                print(f"Warning: Error processing {csv_file.name}: {e}")
                continue

        if not self.timestamps:
            raise ValueError("No valid option chain data with timestamps found")

    def _compute_net_gex_near_spot(self, df):
        """Compute near-spot Gross GEX for strikes within strike_width of spot.

        Args:
            df: DataFrame with option chain data

        Returns:
            tuple: (Gross gamma exposure, number of strikes used)
        """
        if df is None or df.empty:
            return 0.0, 0

        if "underlying_price" not in df.columns:
            return 0.0, 0

        spot = float(df["underlying_price"].iloc[0])

        # Filter to near-spot strikes
        band = df[
            (df["strike"] >= spot - self.strike_width) & (df["strike"] <= spot + self.strike_width)
        ]

        strike_count = len(band)
        gross_gex = self._compute_net_gex(band, spot)
        return gross_gex, strike_count

    def _compute_net_gex(self, df, spot):
        """Compute GROSS gamma exposure for the given option chain slice.

        This is the base metric and does not assume dealer positioning.

        Args:
            df: DataFrame with option chain data (filtered)
            spot: Current underlying price

        Returns:
            float: Gross gamma exposure
        """
        if df is None or df.empty:
            return 0.0

        gross = row_gross_gex(df, spot, self.multiplier, self.gamma_scale)
        return float(gross.sum())

    def _compute_top5_gex(self, df):
        """Compute Gross GEX for the top 5 strikes identified in first file.

        Args:
            df: DataFrame with option chain data

        Returns:
            float: Gross gamma exposure for top 5 strikes
        """
        if df is None or df.empty or not self.top5_strikes:
            return 0.0

        if "underlying_price" not in df.columns:
            return 0.0

        spot = float(df["underlying_price"].iloc[0])

        # Filter to only top 5 strikes
        top5_df = df[df["strike"].isin(self.top5_strikes)]

        if top5_df.empty:
            return 0.0

        gross = row_gross_gex(top5_df, spot, self.multiplier, self.gamma_scale)
        return float(gross.sum())
