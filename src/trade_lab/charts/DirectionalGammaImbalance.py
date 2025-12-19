from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gamma_utils import row_gross_gex


class DirectionalGammaImbalance:
    def __init__(
        self,
        data_dir="data",
        strike_width=50.0,
        multiplier=100.0,
        gamma_scale=0.01,
    ):
        """
        Initialize DirectionalGammaImbalance calculator and plotter.

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
        self.dgi_scores = []
        self.strike_counts = []
        self.top5_dgi_scores = []
        self.top5_strikes = None

    def plot(self, figsize=(14, 7), save_path=None):
        """
        Plot Directional Gamma Imbalance over time.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure

        Returns:
            Tuple of (fig, ax)
        """
        if not self.timestamps:
            raise ValueError("No data to plot. Call load_and_calculate() first.")

        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot DGI on primary axis
        ax1.plot(self.timestamps, self.dgi_scores, "b-", linewidth=2, label="DGI (Strike Window)")
        ax1.scatter(self.timestamps, self.dgi_scores, c="blue", s=20, zorder=5)

        # Plot Top 5 Strikes DGI if available
        if self.top5_dgi_scores:
            ax1.plot(self.timestamps, self.top5_dgi_scores, "r-", linewidth=2, label="DGI (Top 5 OI)", linestyle="--")
            ax1.scatter(self.timestamps, self.top5_dgi_scores, c="red", s=20, zorder=5)

        # Zero line
        ax1.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # Format x-axis to show time as HH:MM
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Labels and styling
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Directional Gamma Imbalance")
        title = f"Intraday Directional Gamma Imbalance (±{self.strike_width} strike window)"
        ax1.set_title(title)
        ax1.set_ylim(-1.0, 1.0)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax1

    def load_and_calculate(self, symbol=None, expiration_filter=None, sample_date=None):
        """Load all option chain CSV files and calculate Directional Gamma Imbalance
        for each timestamp.

        Args:
            symbol: Trading symbol to filter files (e.g., '$SPX', 'SPXW')
            expiration_filter: Expiration date string (YYYY-MM-DD) to filter files (required)
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

                # Calculate DGI for this snapshot
                dgi_norm, strike_count = self._calculate_dgi(df)

                # Calculate DGI for top 5 strikes if identified
                top5_dgi = 0.0
                if self.top5_strikes:
                    top5_dgi = self._calculate_top5_dgi(df)

                self.timestamps.append(timestamp)
                self.dgi_scores.append(dgi_norm)
                self.strike_counts.append(strike_count)
                self.top5_dgi_scores.append(top5_dgi)

            except Exception as e:
                print(f"Warning: Error processing {csv_file.name}: {e}")
                continue

        if not self.timestamps:
            raise ValueError("No valid option chain data with timestamps found")

    def _calculate_dgi(self, df):
        """
        Directional Gamma Imbalance (DGI), computed from GROSS GEX.

        Filters options by strike window around spot, then computes gamma exposure above
        vs below spot to determine directional gamma imbalance.

        Returns:
            tuple: (DGI score in [-1, +1], number of strikes used)
            < 0: more gamma mass above spot than below
            > 0: more gamma mass below spot than above
        """
        if df.empty or "underlying_price" not in df.columns:
            return 0.0, 0

        spot = df["underlying_price"].iloc[0]

        # Filter to near-spot strikes
        filtered_df = df.loc[
            (df["strike"] >= spot - self.strike_width)
            & (df["strike"] <= spot + self.strike_width)
        ].copy()

        if filtered_df.empty:
            return 0.0, 0

        strike_count = len(filtered_df)
        filtered_df["gex"] = row_gross_gex(filtered_df, spot, self.multiplier, self.gamma_scale)

        gamma_above = filtered_df.loc[filtered_df["strike"] > spot, "gex"].sum()
        gamma_below = filtered_df.loc[filtered_df["strike"] < spot, "gex"].sum()

        denom = abs(gamma_above) + abs(gamma_below)
        if denom == 0:
            return 0.0, strike_count

        dgi = (gamma_above - gamma_below) / denom
        return float(np.clip(dgi, -1.0, 1.0)), strike_count

    def _calculate_top5_dgi(self, df):
        """
        Calculate DGI for only the top 5 strikes identified in first file.

        Args:
            df: DataFrame with option chain data

        Returns:
            float: DGI score in [-1, +1] for top 5 strikes
        """
        if df.empty or "underlying_price" not in df.columns or not self.top5_strikes:
            return 0.0

        spot = df["underlying_price"].iloc[0]

        # Filter to only top 5 strikes
        top5_df = df[df["strike"].isin(self.top5_strikes)].copy()

        if top5_df.empty:
            return 0.0

        top5_df["gex"] = row_gross_gex(top5_df, spot, self.multiplier, self.gamma_scale)

        gamma_above = top5_df.loc[top5_df["strike"] > spot, "gex"].sum()
        gamma_below = top5_df.loc[top5_df["strike"] < spot, "gex"].sum()

        denom = abs(gamma_above) + abs(gamma_below)
        if denom == 0:
            return 0.0

        dgi = (gamma_above - gamma_below) / denom
        return float(np.clip(dgi, -1.0, 1.0))
