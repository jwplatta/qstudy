"""GEX Slope chart - Delta of time-weighted net GEX between samples."""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.intraday import load_intraday_option_samples


class GEXSlope:
    """
    Chart class for visualizing the change (slope) in time-weighted net GEX over time.

    For each intraday sample, calculates the net GEX (calls - puts) within
    a ±strike_window around spot, with time-weighting by expiration.
    Then computes the delta between the current sample and a previous sample.

    Formulas:
        weight_e = 1 / sqrt(DTE_e + 1)
        NetGEX_e(t) = Σ_{k in window} NetGEX(e, k, t)
        NetGEX_total(t) = Σ_e weight_e × NetGEX_e(t)
        GEX_delta(t) = NetGEX_total(t) - NetGEX_total(t - lookback)

    Interpretation:
        - GEX delta < 0: GEX deteriorating (pause)
        - GEX delta near 0: stabilization (balance)
        - GEX delta > 0: GEX improving
    """

    def __init__(
        self,
        symbol="SPXW",
        sample_date=None,
        days_out=10,
        strike_window=50,
        lookback=1,
        data_dir="data",
        debug=False,
    ):
        """
        Initialize GEXSlope chart.

        Args:
            symbol: The ticker symbol (e.g., 'SPXW', 'SPX')
            sample_date: Date to load samples for (YYYY-MM-DD string, defaults to today)
            days_out: Number of calendar days to include expirations (default: 10)
            strike_window: Window in points around spot (default: 50 for ±50 pts)
            lookback: Number of samples to look back for delta calculation (default: 1)
            data_dir: Directory containing option chain CSV files
            debug: Verbose output flag
        """
        if days_out > 45:
            raise ValueError("days_out should not exceed 45 days to limit data volume.")

        self.symbol = symbol
        self.sample_date = sample_date or datetime.now().strftime("%Y-%m-%d")
        self.days_out = days_out
        self.strike_window = strike_window
        self.lookback = lookback
        self.data_dir = Path(data_dir)
        self.debug = debug

        self.gex_series = None

    def load_data(self):
        """
        Load option chain samples and calculate time-weighted net GEX window for each.

        For each intraday sample:
        1. Get the spot price at that sample time
        2. For each expiration, calculate net GEX within ±strike_window of spot
        3. Apply time weight: weight_e = 1 / sqrt(DTE_e + 1)
        4. Sum weighted net GEX across all expirations
        5. Calculate delta vs lookback samples prior

        Formula: GEX_delta(t) = NetGEX_total(t) - NetGEX_total(t - lookback)
        """
        samples = load_intraday_option_samples(
            self.symbol, self.sample_date, self.data_dir, self.days_out
        )

        if self.debug:
            print(f"Loaded {len(samples)} intraday samples for {self.sample_date}")

        sample_dt = datetime.strptime(self.sample_date, "%Y-%m-%d")

        # Calculate time-weighted net GEX for each sample
        gex_data = []
        for fetch_dt, df in samples:
            # Get spot price from this sample
            spot = pd.to_numeric(df["underlying_price"], errors="coerce").dropna().iloc[0]

            # Ensure numeric columns
            df = df.copy()
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
            df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")

            # Filter to strike window around spot
            mask = (df["strike"] >= spot - self.strike_window) & (
                df["strike"] <= spot + self.strike_window
            )
            window_df = df[mask].copy()

            if window_df.empty:
                gex_data.append({"datetime": fetch_dt, "net_gex": 0.0, "spot": spot})
                continue

            # Calculate GEX: gamma * OI * spot^2
            window_df["gex"] = window_df["gamma"] * window_df["open_interest"] * (spot**2)

            # Calculate time-weighted net GEX per expiration
            weighted_net_gex = 0.0
            for exp_date, exp_group in window_df.groupby("expiration_date"):
                # Calculate DTE
                exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                dte = (exp_dt - sample_dt).days

                # Time weight: 1 / sqrt(DTE + 1)
                weight = 1.0 / np.sqrt(dte + 1)

                # Net GEX for this expiration = calls - puts
                calls = exp_group[exp_group["contract_type"] == "CALL"]["gex"].sum()
                puts = exp_group[exp_group["contract_type"] == "PUT"]["gex"].sum()
                net_gex_exp = calls - puts

                weighted_net_gex += weight * net_gex_exp

                if self.debug:
                    print(
                        f"    Exp {exp_date} (DTE={dte}): weight={weight:.3f}, "
                        f"net_gex={net_gex_exp:,.0f}, weighted={weight * net_gex_exp:,.0f}"
                    )

            gex_data.append({"datetime": fetch_dt, "net_gex": weighted_net_gex, "spot": spot})

            if self.debug:
                print(
                    f"  {fetch_dt.strftime('%H:%M:%S')}: Spot = {spot:.1f}, "
                    f"Weighted Net GEX (±{self.strike_window}pts) = {weighted_net_gex:,.0f}"
                )

        self.gex_series = pd.DataFrame(gex_data).set_index("datetime").sort_index()

        # Calculate delta (slope) between samples
        self.gex_series["gex_delta"] = self.gex_series["net_gex"] - self.gex_series[
            "net_gex"
        ].shift(self.lookback)

    def plot(self, figsize=(12, 6), save_path=None):
        """
        Generate and display the GEX delta (slope) over time plot.

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the figure

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if self.gex_series is None:
            self.load_data()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot the delta as a line graph
        gex_delta = self.gex_series["gex_delta"].dropna()

        ax.plot(
            gex_delta.index,
            gex_delta.values,
            linewidth=2,
            color="steelblue",
            marker="o",
            markersize=4,
        )

        # Add zero reference line
        ax.axhline(0, linestyle="--", color="black", alpha=0.5, label="Zero")

        # Format x-axis for intraday with 30-minute ticks starting at 8:30
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.set_xlim(left=pd.to_datetime(f"{self.sample_date} 08:30"))
        plt.xticks(rotation=45)

        ax.set_xlabel("Time")
        ax.set_ylabel(f"GEX Delta (±{self.strike_window}pts)")
        ax.set_title(
            f"{self.symbol} GEX Slope ({self.sample_date}) - "
            f"{self.days_out}d expirations, ±{self.strike_window}pt window"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            if save_path is True:
                save_path = f"{self.symbol}_gex_slope_{self.sample_date}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
