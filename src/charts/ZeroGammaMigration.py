"""Zero-Gamma Migration chart for structural center analysis."""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from utils.intraday import calculate_zero_gamma_line, load_intraday_option_samples


class ZeroGammaMigration:
    """
    Chart class for visualizing zero-gamma line migration over time.

    Detects whether the structural center of the market is still moving:
    - > ~10-15 pts/hr: migration (pause)
    - < ~5 pts/hr: anchored (balance)

    Balance requires zero-gamma to anchor.
    """

    def __init__(
        self,
        symbol="SPXW",
        sample_date=None,
        days_out=10,
        lookback=1,
        data_dir="data",
        debug=False,
    ):
        """
        Initialize ZeroGammaMigration chart.

        Args:
            symbol: The ticker symbol (e.g., 'SPXW', 'SPX')
            sample_date: Date to load samples for (YYYY-MM-DD string, defaults to today)
            days_out: Number of calendar days to include expirations (default: 10)
            lookback: Number of samples to look back for delta calculation (default: 1)
            data_dir: Directory containing option chain CSV files
            debug: Verbose output flag
        """
        self.symbol = symbol
        self.sample_date = sample_date or datetime.now().strftime("%Y-%m-%d")
        self.days_out = days_out
        self.lookback = lookback
        self.data_dir = Path(data_dir)
        self.debug = debug

        self.zgl_series = None

    def load_data(self):
        """
        Load option chain samples and calculate zero-gamma line for each.

        Uses the GEXPrice algorithm to find where net GEX crosses zero
        for each intraday sample.
        """
        samples = load_intraday_option_samples(
            self.symbol, self.sample_date, self.data_dir, self.days_out
        )

        if self.debug:
            print(f"Loaded {len(samples)} intraday samples for {self.sample_date}")

        # Calculate zero-gamma line for each sample
        zgl_data = []
        for fetch_dt, df in samples:
            # Get spot price
            spot = pd.to_numeric(df["underlying_price"], errors="coerce").dropna().iloc[0]

            # Calculate zero-gamma line
            zgl = calculate_zero_gamma_line(df, spot, self.days_out)

            if zgl is not None:
                zgl_data.append({"datetime": fetch_dt, "zgl": zgl, "spot": spot})

                if self.debug:
                    print(
                        f"  {fetch_dt.strftime('%H:%M:%S')}: ZGL = {zgl:.1f}, "
                        f"Spot = {spot:.1f}, Diff = {zgl - spot:.1f}"
                    )

        if not zgl_data:
            raise ValueError(f"No valid zero-gamma line data for {self.sample_date}")

        self.zgl_series = pd.DataFrame(zgl_data).set_index("datetime").sort_index()

        # Calculate delta (migration) between samples
        self.zgl_series["zgl_delta"] = self.zgl_series["zgl"] - self.zgl_series["zgl"].shift(
            self.lookback
        )

    def plot(self, figsize=(12, 8), save_path=None):
        """
        Generate and display the zero-gamma migration plot.

        Creates a 2-panel plot:
        - Top: ZGL level vs spot over time
        - Bottom: ZGL delta (migration rate) with zero reference line

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the figure

        Returns:
            tuple: (fig, axes) matplotlib figure and axes objects
        """
        if self.zgl_series is None:
            self.load_data()

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Top panel: ZGL level vs spot
        ax1 = axes[0]
        ax1.plot(
            self.zgl_series.index,
            self.zgl_series["zgl"],
            linewidth=2,
            color="blue",
            marker="o",
            markersize=4,
            label="Zero-Gamma Line",
        )
        ax1.plot(
            self.zgl_series.index,
            self.zgl_series["spot"],
            linewidth=2,
            color="gray",
            linestyle="--",
            marker="s",
            markersize=3,
            label="Spot",
        )
        ax1.set_ylabel("Price Level")
        ax1.set_title(
            f"{self.symbol} Zero-Gamma Line ({self.sample_date}) - {self.days_out}d window"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom panel: ZGL delta (migration rate)
        ax2 = axes[1]
        zgl_delta = self.zgl_series["zgl_delta"].dropna()

        # Color bars based on magnitude (red for large moves)
        colors = []
        for x in zgl_delta.values:
            if abs(x) > 10:
                colors.append("red")  # Significant migration
            elif abs(x) > 5:
                colors.append("orange")  # Moderate migration
            else:
                colors.append("green")  # Stable/anchored

        ax2.bar(zgl_delta.index, zgl_delta.values, width=0.002, color=colors, alpha=0.7)
        ax2.axhline(0, linestyle="-", color="black", linewidth=1)

        # Add threshold lines
        ax2.axhline(10, linestyle=":", color="red", alpha=0.5, label="+10 pts (migration)")
        ax2.axhline(-10, linestyle=":", color="red", alpha=0.5)
        ax2.axhline(5, linestyle=":", color="orange", alpha=0.5)
        ax2.axhline(-5, linestyle=":", color="orange", alpha=0.5)

        ax2.set_ylabel(f"ZGL Delta (vs {self.lookback} sample(s) prior)")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)

        # Add interpretation annotation
        ax2.annotate(
            ">10 pts/sample = Migration (Pause)\n<5 pts/sample = Anchored (Balance)",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Format x-axis for intraday
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)

        fig.tight_layout()

        if save_path:
            if save_path is True:
                save_path = f"{self.symbol}_zgl_migration_{self.sample_date}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes
