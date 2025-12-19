from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


class NetGEX:
    """Net Gamma Exposure (NET GEX) calculator and plotter.

    Computes and visualizes net gamma exposure over time to identify
    market regime (mean reversion vs trend/breakout).

    NET GEX > 0: Dealers long gamma -> hedging dampens moves (mean reversion)
    NET GEX < 0: Dealers short gamma -> hedging amplifies moves (trend risk)
    """

    def __init__(
        self,
        data_dir="data",
        strike_width=50.0,
        multiplier=100.0,
        dealer_short=True,
        gamma_scale=0.01,
    ):
        """Initialize NetGEX calculator and plotter.

        Args:
            data_dir: Directory containing option chain CSV files
            strike_width: Half-width of strike band around spot (default: 50.0)
                         e.g., 50 means strikes within +/- 50 points of spot
            multiplier: Contract multiplier (default: 100.0 for SPX)
            dealer_short: If True, dealers are short options (sign=-1)
            gamma_scale: Scaling factor for gamma units (default: 0.01)
        """
        self.data_dir = Path(data_dir)
        self.strike_width = strike_width
        self.multiplier = multiplier
        self.dealer_short = dealer_short
        self.gamma_scale = gamma_scale
        self.timestamps = []
        self.net_gex_values = []

    def plot(self, figsize=(14, 7), save_path=None):
        """Plot Net Gamma Exposure over time as a line chart.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure

        Returns:
            Tuple of (fig, ax)
        """
        if not self.timestamps:
            raise ValueError("No data to plot. Call load_and_calculate() first.")

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.timestamps, self.net_gex_values, "b-", linewidth=2)
        ax.scatter(self.timestamps, self.net_gex_values, c="blue", s=20, zorder=5)

        # Zero line
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # Format x-axis to show time as HH:MM
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Labels and styling
        ax.set_xlabel("Time")
        ax.set_ylabel("Net Gamma Exposure")
        title = f"Intraday Net GEX (±{self.strike_width} strike window)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def load_and_calculate(self, symbol=None, expiration_filter=None, sample_date=None):
        """Load all option chain CSV files and calculate Net GEX for each timestamp.

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

                # Calculate Net GEX for this snapshot
                net_gex = self._compute_net_gex_near_spot(df)

                self.timestamps.append(timestamp)
                self.net_gex_values.append(net_gex)

            except Exception as e:
                print(f"Warning: Error processing {csv_file.name}: {e}")
                continue

        if not self.timestamps:
            raise ValueError("No valid option chain data with timestamps found")

    def _compute_net_gex_near_spot(self, df):
        """Compute near-spot NET GEX for strikes within strike_width of spot.

        Args:
            df: DataFrame with option chain data

        Returns:
            float: Net gamma exposure
        """
        if df is None or df.empty:
            return 0.0

        if "underlying_price" not in df.columns:
            return 0.0

        spot = float(df["underlying_price"].iloc[0])

        # Filter to near-spot strikes
        band = df[
            (df["strike"] >= spot - self.strike_width) & (df["strike"] <= spot + self.strike_width)
        ]

        return self._compute_net_gex(band, spot)

    def _compute_net_gex(self, df, spot):
        """Compute NET gamma exposure for the given option chain slice.

        Args:
            df: DataFrame with option chain data (filtered)
            spot: Current underlying price

        Returns:
            float: Net gamma exposure
        """
        if df is None or df.empty:
            return 0.0

        sign = -1.0 if self.dealer_short else 1.0

        gex = (
            sign
            * df["gamma"]
            * df["open_interest"]
            * (spot**2)
            * self.multiplier
            * self.gamma_scale
        )

        return float(gex.sum())
