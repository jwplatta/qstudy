from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GEX:
    """Gamma Exposure (GEX) charting utilities.

    This class provides methods to calculate and visualize gamma exposure
    from SPX option chain data.
    """

    MULTIPLIER = 100  # Standard options contract multiplier

    def __init__(
        self, symbol=None, expiration_date=None, data_dir="data", csv_path=None, dataframe=None
    ):
        """Initialize GEX chart.

        Args:
            symbol: Trading symbol (e.g., '$SPX', 'SPXW')
            expiration_date: Option expiration date in YYYY-MM-DD format
            data_dir: Directory containing option chain CSV files
            csv_path: Path to CSV file containing option chain data (legacy)
            dataframe: Pandas DataFrame with option chain data (legacy)
        """
        self.data_dir = Path(data_dir)
        self.df = None

        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        elif dataframe is not None:
            self.df = dataframe.copy()
        elif symbol is not None and expiration_date is not None:
            self._load_data(symbol, expiration_date)
        else:
            raise ValueError(
                "Must provide either (symbol and expiration_date) or csv_path or dataframe"
            )

        self._prepare_data()

    def plot(self, min_strike=None, max_strike=None, date=None, figsize=(10, 6)):
        """Plot gamma exposure chart.

        Args:
            min_strike: Minimum strike to display
            max_strike: Maximum strike to display
            date: Date string for chart title (optional)
            figsize: Figure size tuple (width, height)
        """
        # Calculate GEX by strike
        gex_filtered = self.calculate_gex_by_strike(min_strike, max_strike)

        # Find zero gamma level
        zero_gamma_strike = self.find_zero_gamma_level(gex_filtered)

        # Get underlying price
        underlying_price = self.df["underlying_price"].iloc[0]

        # Calculate totals
        total_call_gex = gex_filtered["CALL"].sum()
        total_put_gex = gex_filtered["PUT"].sum()

        # Print summary statistics
        strike_range = f"{min_strike or gex_filtered['strike'].min():.0f}-{max_strike or gex_filtered['strike'].max():.0f}"
        print(f"Strike range: {strike_range}")
        print(f"Total Call Gamma Exposure: {total_call_gex:,.0f}")
        print(f"Total Put Gamma Exposure:  {total_put_gex:,.0f}")
        if zero_gamma_strike:
            print(f"Zero Gamma Level: ≈ {zero_gamma_strike:.1f}")

        # Create plot
        fig, ax1 = plt.subplots(figsize=figsize)

        # Set x-axis ticks
        if min_strike is not None and max_strike is not None:
            ax1.set_xticks(np.arange(min_strike, max_strike + 1, 20))

        # Bar chart for calls and puts
        ax1.bar(
            gex_filtered["strike"],
            gex_filtered["CALL"],
            width=5,
            label="Calls",
            color="steelblue",
            alpha=0.8,
        )
        ax1.bar(
            gex_filtered["strike"],
            gex_filtered["PUT"],
            width=5,
            label="Puts",
            color="orange",
            alpha=0.7,
        )

        # Net gamma line (right axis)
        ax2 = ax1.twinx()
        ax2.plot(
            gex_filtered["strike"],
            gex_filtered["net_gamma"],
            color="black",
            lw=2,
            label="Net Gamma (Calls - Puts)",
        )

        # Underlying price line
        ax1.axvline(
            underlying_price,
            color="gray",
            linestyle="--",
            lw=1.5,
            label=f"Underlying ({underlying_price:.1f})",
        )

        # Zero gamma line
        if zero_gamma_strike:
            ax1.axvline(
                zero_gamma_strike,
                color="red",
                linestyle="--",
                lw=1.2,
                label=f"Zero Gamma ≈ {zero_gamma_strike:.1f}",
            )

        # Labels and styling
        title = f"SPX Gamma Exposure ({strike_range} strikes)"
        if date:
            title += f" - {date}"
        ax1.set_title(title)
        ax1.set_xlabel("Strike Price")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_ylabel("Gamma Exposure ($ per 1pt move)")
        ax2.set_ylabel("Net Gamma Exposure ($)")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        return fig, (ax1, ax2)

    def _prepare_data(self):
        """Ensure numeric columns are parsed properly."""
        numeric_columns = ["strike", "gamma", "open_interest", "underlying_price"]
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df["gex"] = (
            self.df["gamma"]
            * self.df["open_interest"]
            * self.MULTIPLIER
            * self.df["underlying_price"]
        )

    def calculate_gex_by_strike(self, min_strike=None, max_strike=None):
        """Calculate gamma exposure aggregated by strike price.

        Args:
            min_strike: Minimum strike to include (optional)
            max_strike: Maximum strike to include (optional)

        Returns:
            DataFrame with columns: strike, CALL, PUT, net_gamma
        """
        # Aggregate by strike and contract type
        gex_by_strike = (
            self.df.groupby(["strike", "contract_type"])["gex"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Filter strike range if specified
        if min_strike is not None or max_strike is not None:
            mask = pd.Series([True] * len(gex_by_strike))
            if min_strike is not None:
                mask = mask & (gex_by_strike["strike"] >= min_strike)
            if max_strike is not None:
                mask = mask & (gex_by_strike["strike"] <= max_strike)

            gex_by_strike = gex_by_strike[mask].copy()

        # Compute net gamma (calls - puts)
        gex_by_strike["net_gamma"] = gex_by_strike["CALL"] - gex_by_strike["PUT"]

        return gex_by_strike

    def find_zero_gamma_level(self, gex_data):
        """Find the strike price where net gamma crosses zero.

        Args:
            gex_data: DataFrame from calculate_gex_by_strike()

        Returns:
            Float representing zero gamma strike, or None if not found
        """
        sign_changes = np.where(np.sign(gex_data["net_gamma"]).diff() != 0)[0]

        if len(sign_changes) > 0:
            idx = sign_changes[0]
            zero_gamma_strike = np.interp(
                0,
                [gex_data["net_gamma"].iloc[idx], gex_data["net_gamma"].iloc[idx + 1]],
                [gex_data["strike"].iloc[idx], gex_data["strike"].iloc[idx + 1]],
            )
            return zero_gamma_strike

        return None

    def _load_data(self, symbol, expiration_date):
        """Load the most recent option chain snapshot for a given symbol and expiration.

        Args:
            symbol: Trading symbol (e.g., '$SPX', 'SPXW')
            expiration_date: Option expiration date in YYYY-MM-DD format
        """
        # Find all CSV files for this symbol and expiration
        pattern = f"{symbol}_exp{expiration_date}_*.csv"
        csv_files = sorted(self.data_dir.glob(pattern))

        if not csv_files:
            raise ValueError(
                f"No option chain CSV files found for symbol {symbol} with expiration {expiration_date} in {self.data_dir}"
            )

        # Get the most recent file (sorted alphabetically by timestamp)
        latest_file = csv_files[-1]
        self.df = pd.read_csv(latest_file)
