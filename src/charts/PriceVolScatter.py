"""Price vs Vol Scatter chart for regime classification."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.intraday import find_closest_expiration, get_atm_iv


class PriceVolScatter:
    """
    Chart class for visualizing price change vs volatility change scatter.

    Directly visualizes pause vs balance regimes:
    - (+dPrice, -dVol): balance / normalization
    - (-dPrice, +dVol): impulse / pause
    - (0, +dVol): hidden stress
    - (0, -dVol): true balance
    """

    def __init__(
        self,
        symbol="SPXW",
        sample_date=None,
        target_dte=7,
        diff_periods=1,
        data_dir="data",
        debug=False,
    ):
        """
        Initialize PriceVolScatter chart.

        Args:
            symbol: The ticker symbol (e.g., 'SPXW', 'SPX')
            sample_date: Date to load samples for (YYYY-MM-DD string, defaults to today)
            target_dte: Target days to expiration for vol (default: 7)
            diff_periods: Number of periods for diff calculation (default: 1)
            data_dir: Directory containing option chain CSV files
            debug: Verbose output flag
        """
        self.symbol = symbol
        self.sample_date = sample_date or datetime.now().strftime("%Y-%m-%d")
        self.target_dte = target_dte
        self.diff_periods = diff_periods
        self.data_dir = Path(data_dir)
        self.debug = debug

        self.expiration = None
        self.price_vol_series = None

    def load_data(self):
        """
        Load option chain data and extract price and ATM IV for each sample.

        Uses the same logic as FrontWeekATMIV to get ATM IV, and also
        extracts the underlying price for each sample.
        """
        # Find closest expiration to target_dte
        self.expiration = find_closest_expiration(
            self.sample_date, self.target_dte, self.data_dir, self.symbol
        )

        if self.debug:
            print(f"Using expiration {self.expiration} (target DTE: {self.target_dte})")

        # Find all files for this expiration sampled on sample_date
        pattern = f"{self.symbol}_exp{self.expiration}_{self.sample_date}_*.csv"
        csv_files = sorted(self.data_dir.glob(pattern))

        if not csv_files:
            raise ValueError(
                f"No option chain files found for {self.symbol} "
                f"exp {self.expiration} on {self.sample_date}"
            )

        if self.debug:
            print(f"Found {len(csv_files)} samples for {self.expiration}")

        # Extract price and ATM IV for each sample
        data = []
        for csv_file in csv_files:
            try:
                parts = csv_file.stem.split("_")
                if len(parts) >= 4:
                    sample_time = parts[3]
                    fetch_dt = datetime.strptime(
                        f"{self.sample_date}_{sample_time}", "%Y-%m-%d_%H-%M-%S"
                    )

                    df = pd.read_csv(csv_file)

                    # Get spot price
                    spot = pd.to_numeric(df["underlying_price"], errors="coerce").dropna().iloc[0]

                    # Get ATM IV
                    atm_iv = get_atm_iv(df)

                    if pd.notna(atm_iv):
                        data.append({"datetime": fetch_dt, "price": spot, "atm_iv": atm_iv})

                        if self.debug:
                            print(
                                f"  {sample_time}: Price = {spot:.2f}, ATM IV = {atm_iv * 100:.2f}%"
                            )
            except Exception as e:
                if self.debug:
                    print(f"Error processing {csv_file}: {e}")
                continue

        if not data:
            raise ValueError(f"No valid price/IV data extracted for {self.sample_date}")

        self.price_vol_series = pd.DataFrame(data).set_index("datetime").sort_index()

        # Calculate differences
        self.price_vol_series["dPrice"] = self.price_vol_series["price"].diff(self.diff_periods)
        self.price_vol_series["dVol"] = self.price_vol_series["atm_iv"].diff(self.diff_periods)

    def plot(self, figsize=(8, 8), save_path=None, colorby="time"):
        """
        Generate and display the price vs vol scatter plot.

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the figure
            colorby: How to color points - 'time' for temporal gradient,
                     'quadrant' for regime classification

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if self.price_vol_series is None:
            self.load_data()

        plot_df = self.price_vol_series.dropna(subset=["dPrice", "dVol"])

        if plot_df.empty:
            raise ValueError("No valid diff data to plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Convert IV diff to percentage points for readability
        dvol_pct = plot_df["dVol"] * 100

        if colorby == "time":
            # Color by time (earlier = lighter)
            scatter = ax.scatter(
                plot_df["dPrice"],
                dvol_pct,
                c=range(len(plot_df)),
                cmap="viridis",
                s=50,
                alpha=0.7,
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Sample Order (Early -> Late)")
        elif colorby == "quadrant":
            # Color by quadrant
            colors = []
            for dp, dv in zip(plot_df["dPrice"], dvol_pct):
                if dp > 0 and dv < 0:
                    colors.append("green")  # Balance/normalization
                elif dp < 0 and dv > 0:
                    colors.append("red")  # Impulse/pause
                elif dp == 0 and dv > 0:
                    colors.append("orange")  # Hidden stress
                elif dp == 0 and dv < 0:
                    colors.append("blue")  # True balance
                else:
                    colors.append("gray")  # Mixed/transition

            ax.scatter(plot_df["dPrice"], dvol_pct, c=colors, s=50, alpha=0.7)

        # Add quadrant lines
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        # Add quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        pad_x = (xlim[1] - xlim[0]) * 0.05
        pad_y = (ylim[1] - ylim[0]) * 0.05

        ax.text(
            xlim[1] - pad_x,
            ylim[0] + pad_y,
            "Balance\nDrift",
            ha="right",
            va="bottom",
            fontsize=10,
            color="green",
            alpha=0.7,
        )
        ax.text(
            xlim[0] + pad_x,
            ylim[1] - pad_y,
            "Pause\nStress",
            ha="left",
            va="top",
            fontsize=10,
            color="red",
            alpha=0.7,
        )
        ax.text(
            xlim[0] + pad_x,
            ylim[0] + pad_y,
            "Balance\nNormalization",
            ha="left",
            va="bottom",
            fontsize=10,
            color="blue",
            alpha=0.7,
        )
        ax.text(
            xlim[1] - pad_x,
            ylim[1] - pad_y,
            "Drift\nStress",
            ha="right",
            va="top",
            fontsize=10,
            color="orange",
            alpha=0.7,
        )

        # Calculate DTE for title
        sample_dt = datetime.strptime(self.sample_date, "%Y-%m-%d")
        exp_dt = datetime.strptime(self.expiration, "%Y-%m-%d")
        actual_dte = (exp_dt - sample_dt).days

        ax.set_xlabel(f"dPrice ({self.diff_periods}-sample diff, points)")
        ax.set_ylabel(f"dVol ({self.diff_periods}-sample diff, % pts)")
        ax.set_title(
            f"{self.symbol} Price vs Vol Scatter ({self.sample_date})\n"
            f"IV from {self.expiration} ({actual_dte}DTE)"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            if save_path is True:
                save_path = f"{self.symbol}_price_vol_scatter_{self.sample_date}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
