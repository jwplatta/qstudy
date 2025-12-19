from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DirectionalGammaImbalance:
    def __init__(self, data_dir="data", min_abs_delta=0.15, max_abs_delta=0.55):
        """
        Initialize DirectionalGammaImbalance calculator and plotter.

        Args:
            data_dir: Directory containing option chain CSV files
            min_abs_delta: Minimum absolute delta for filtering (default 0.15)
            max_abs_delta: Maximum absolute delta for filtering (default 0.55)
        """
        self.data_dir = Path(data_dir)
        self.min_abs_delta = min_abs_delta
        self.max_abs_delta = max_abs_delta
        self.timestamps = []
        self.dgi_scores = []

    def plot(self, figsize=(14, 7), save_path=None):
        """
        Plot Directional Gamma Imbalance over time as a simple line chart.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure

        Returns:
            Tuple of (fig, ax)
        """
        if not self.timestamps:
            raise ValueError("No data to plot. Call load_and_calculate() first.")

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.timestamps, self.dgi_scores, "b-", linewidth=2)
        ax.scatter(self.timestamps, self.dgi_scores, c="blue", s=20, zorder=5)

        # Zero line
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # Format x-axis to show time as HH:MM
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Labels and styling
        ax.set_xlabel("Time")
        ax.set_ylabel("Directional Gamma Imbalance")
        ax.set_title("Intraday Directional Gamma Imbalance")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

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
                dgi_norm = self._calculate_dgi(df)

                self.timestamps.append(timestamp)
                self.dgi_scores.append(dgi_norm)

            except Exception as e:
                print(f"Warning: Error processing {csv_file.name}: {e}")
                continue

        if not self.timestamps:
            raise ValueError("No valid option chain data with timestamps found")

    def _calculate_dgi(self, df):
        """
        Calculate Directional Gamma Imbalance (DGI).

        Filters options by delta range, then computes gamma exposure above
        vs  belowspot to determine directional gamma imbalance.

        Returns:
            float in [-1, +1]
            < 0: upside fragile
            > 0: downside fragile
        """
        if df.empty or "underlying_price" not in df.columns:
            return 0.0

        spot = df["underlying_price"].iloc[0]

        # Filter by delta range
        filtered_df = df.loc[
            (df["delta"].abs() >= self.min_abs_delta) & (df["delta"].abs() <= self.max_abs_delta)
        ].copy()

        if filtered_df.empty:
            return 0.0

        # Calculate gamma exposure (dealers are short options)
        filtered_df["gex"] = -filtered_df["gamma"] * filtered_df["open_interest"] * (spot**2) * 0.01

        # Sum gamma above and below spot
        gamma_above = filtered_df.loc[filtered_df["strike"] > spot, "gex"].sum()
        gamma_below = filtered_df.loc[filtered_df["strike"] < spot, "gex"].sum()

        denom = abs(gamma_above) + abs(gamma_below)

        if denom == 0:
            return 0.0

        dgi = (gamma_above - gamma_below) / denom

        # Clamp to [-1, +1] range
        return np.clip(dgi, -1.0, 1.0)
