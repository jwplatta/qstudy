import matplotlib.pyplot as plt
import numpy as np

from .GEXStrike import GEXStrike


class CompareGEX:
    """Compare Gamma Exposure (GEX) across multiple expiration dates.

    This class creates a grid of GEX charts, one for each expiration date,
    to facilitate comparison of gamma exposure patterns across different
    time horizons.
    """

    def __init__(self, symbol, expiration_dates, data_dir="data"):
        """Initialize CompareGEX chart.

        Args:
            symbol: Trading symbol (e.g., '$SPX', 'SPXW')
            expiration_dates: List of expiration date strings in YYYY-MM-DD format
            data_dir: Directory containing option chain CSV files
        """
        self.symbol = symbol
        self.expiration_dates = sorted(expiration_dates)
        self.data_dir = data_dir

    def plot(self, min_strike=None, max_strike=None, figsize=None):
        """Plot GEX comparison chart with subplots for each expiration.

        Args:
            min_strike: Minimum strike to display across all charts
            max_strike: Maximum strike to display across all charts
            figsize: Figure size tuple (width, height). If None, auto-calculated

        Returns:
            Tuple of (figure, axes_array)
        """
        n_charts = len(self.expiration_dates)

        # Calculate grid dimensions (max 2 columns)
        n_cols = min(2, n_charts)
        n_rows = int(np.ceil(n_charts / n_cols))

        # Auto-calculate figsize if not provided
        if figsize is None:
            figsize = (n_cols * 10, n_rows * 6)

        # Create subplot grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Ensure axes is always a flat array for consistent iteration
        if n_charts == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        # Plot each expiration date
        for idx, expiry in enumerate(self.expiration_dates):
            try:
                # Create GEX instance for this expiration (it handles data loading)
                gex = GEXStrike(symbol=self.symbol, expiration_date=expiry, data_dir=self.data_dir)
            except ValueError as e:
                print(f"Warning: {e}")
                axes[idx].axis("off")
                continue

            # Calculate GEX by strike
            gex_filtered = gex.calculate_gex_by_strike(min_strike, max_strike)

            # Find zero gamma level
            zero_gamma_strike = gex.find_zero_gamma_level(gex_filtered)

            # Get underlying price
            underlying_price = gex.df["underlying_price"].iloc[0]

            # Plot on this subplot
            ax1 = axes[idx]

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
                label="Net Gamma",
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
            strike_range = f"{min_strike or gex_filtered['strike'].min():.0f}-{max_strike or gex_filtered['strike'].max():.0f}"
            ax1.set_title(f"SPX GEX - Expiry: {expiry}")
            ax1.set_xlabel("Strike Price")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.set_ylabel("Gamma Exposure ($ per 1pt move)")
            ax2.set_ylabel("Net Gamma ($)")
            ax1.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
            ax1.grid(True, linestyle="--", alpha=0.5)

        # Hide any unused subplots
        for idx in range(n_charts, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()

        return fig, axes
