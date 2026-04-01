"""Front-week ATM IV chart for short-dated volatility regime analysis."""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from ..utils.intraday import find_closest_expiration, get_atm_iv


class FrontWeekATMIV:
    """
    Chart class for visualizing front-week ATM implied volatility over time.

    Measures short-dated volatility regime to distinguish balance vs pause:
    - Vol rising: stress / pause
    - Vol flat or falling on flat price: balance
    """

    def __init__(
        self,
        symbol="SPXW",
        sample_date=None,
        target_dte=7,
        data_dir="data",
        debug=False,
    ):
        """
        Initialize FrontWeekATMIV chart.

        Args:
            symbol: The ticker symbol (e.g., 'SPXW', 'SPX')
            sample_date: Date to load samples for (YYYY-MM-DD string, defaults to today)
            target_dte: Target days to expiration (default: 7)
            data_dir: Directory containing option chain CSV files
            debug: Verbose output flag
        """
        self.symbol = symbol
        self.sample_date = sample_date or datetime.now().strftime("%Y-%m-%d")
        self.target_dte = target_dte
        self.data_dir = Path(data_dir)
        self.debug = debug

        self.expiration = None
        self.iv_series = None

    def load_data(self):
        """
        Load option chain data and calculate ATM IV for each intraday sample.

        Finds the expiration closest to target_dte, then loads all samples
        for that expiration and extracts ATM IV for each sample time.
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

        # Extract ATM IV for each sample
        iv_data = []
        for csv_file in csv_files:
            try:
                parts = csv_file.stem.split("_")
                if len(parts) >= 4:
                    sample_time = parts[3]
                    fetch_dt = datetime.strptime(
                        f"{self.sample_date}_{sample_time}", "%Y-%m-%d_%H-%M-%S"
                    )

                    df = pd.read_csv(csv_file)
                    atm_iv = get_atm_iv(df)

                    if pd.notna(atm_iv):
                        iv_data.append({"datetime": fetch_dt, "atm_iv": atm_iv})

                        if self.debug:
                            print(f"  {sample_time}: ATM IV = {atm_iv * 100:.2f}%")
            except Exception as e:
                if self.debug:
                    print(f"Error processing {csv_file}: {e}")
                continue

        if not iv_data:
            raise ValueError(f"No valid ATM IV data extracted for {self.sample_date}")

        self.iv_series = pd.DataFrame(iv_data).set_index("datetime").sort_index()

    def plot(self, figsize=(12, 6), save_path=None):
        """
        Generate and display the ATM IV plot.

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the figure

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if self.iv_series is None:
            self.load_data()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot IV as percentage
        iv_pct = self.iv_series["atm_iv"] * 100
        ax.plot(iv_pct.index, iv_pct.values, linewidth=2, color="purple", marker="o", markersize=4)

        iv_mean = iv_pct.mean()
        ax.axhline(
            iv_mean,
            linestyle="--",
            color="gray",
            alpha=0.7,
            label=f"Mean IV = {iv_mean:.1f}%",
        )

        sample_dt = datetime.strptime(self.sample_date, "%Y-%m-%d")
        exp_dt = datetime.strptime(self.expiration, "%Y-%m-%d")
        actual_dte = (exp_dt - sample_dt).days

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)

        ax.set_xlabel("Time")
        ax.set_ylabel("ATM IV (%)")
        ax.set_title(
            f"{self.symbol} Front-Week ATM IV ({self.sample_date}) - "
            f"Exp {self.expiration} ({actual_dte}DTE)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            if save_path is True:
                save_path = f"{self.symbol}_atm_iv_{self.sample_date}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
