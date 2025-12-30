from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.black_scholes import bs_gamma


class GEXPrice:
    MULTIPLIER = 100

    def __init__(self, symbol=None, start_date=None, days_out=10, data_dir="data", debug=False):
        """
        Initialize GEXPrice chart.

        Args:
            symbol: The ticker symbol (e.g., 'SPXW', 'SPX')
            start_date: Start date as string 'YYYY-MM-DD' (defaults to today)
            days_out: Number of calendar days to include expirations (default: 10)
            data_dir: Directory containing option chain CSV files
        """
        if days_out > 45:
            raise ValueError("days_out should not exceed 45 days to limit data volume.")

        self.symbol = symbol
        self.start_date = start_date or datetime.now().strftime("%Y-%m-%d")
        self.days_out = days_out
        self.data_dir = Path(data_dir)
        self.debug = debug

        self.all_opts = None
        self.spot = None
        self.asof = None

    def load_data(self):
        """Load option chain data for expirations within the date range."""
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=self.days_out)

        months = [start_dt.month, (start_dt.month % 12) + 1]
        years = [start_dt.year, start_dt.year if start_dt.month < 12 else start_dt.year + 1]

        csv_files = []
        for m, y in zip(months, years):
            month_str = f"{y:04d}-{m:02d}"
            pattern = f"{self.symbol}_exp{month_str}*.csv"
            csv_files.extend(sorted(self.data_dir.glob(pattern)))

        if not csv_files:
            raise ValueError(
                f"No option chain CSV files found for symbol {self.symbol} in {self.data_dir}"
            )

        files_by_expiration = {}
        for csv_file in csv_files:
            try:
                parts = csv_file.stem.split("_")
                if len(parts) >= 4:
                    exp_date_str = parts[1].replace("exp", "")
                    exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")

                    fetch_date = parts[2]
                    fetch_time = parts[3]
                    fetch_dt = datetime.strptime(f"{fetch_date}_{fetch_time}", "%Y-%m-%d_%H-%M-%S")

                    if start_dt <= exp_date <= end_dt:
                        if (
                            exp_date_str not in files_by_expiration
                            or fetch_dt > files_by_expiration[exp_date_str][0]
                        ):
                            files_by_expiration[exp_date_str] = (fetch_dt, csv_file)
            except Exception:
                continue

        if not files_by_expiration:
            raise ValueError(
                f"No option chain files found with expirations between {self.start_date} and {end_dt.strftime('%Y-%m-%d')}"
            )

        latest_files = [file_info[1] for file_info in files_by_expiration.values()]
        self.asof = min(file_info[0] for file_info in files_by_expiration.values())

        if self.debug:
            print(f"Loading {len(latest_files)} option chain files for GEX by Price calculation.")
            print(f"Data as of {self.asof.strftime('%Y-%m-%d %H:%M:%S')}")

        dfs = []
        for csv_file in latest_files:
            if self.debug:
                print(csv_file)

            df_temp = pd.read_csv(csv_file)
            if not df_temp.empty:
                dfs.append(df_temp)

        self.all_opts = pd.concat(dfs, ignore_index=True).copy()

        # NOTE: convert the expiration date to a datetime when trading stop:
        # Add 15 hours and 15 minutes.
        # Assumes 3 PM CT expiry.
        self.all_opts["expiration_dt"] = pd.to_datetime(
            self.all_opts["expiration_date"]
        ) + pd.Timedelta(hours=15, minutes=15)

        # Time to expiry in years, floored at ~1 minute
        self.all_opts["T"] = (self.all_opts["expiration_dt"] - self.asof).dt.total_seconds() / (
            365.0 * 24 * 3600
        )
        self.all_opts["T"] = self.all_opts["T"].clip(lower=(5.0 / (365.0 * 24 * 60)))

        # IV: use theoretical_volatility, convert percent -> decimal
        if "theoretical_volatility" not in self.all_opts.columns:
            raise ValueError("Expected theoretical_volatility column for IV input.")

        self.all_opts["iv"] = (
            pd.to_numeric(self.all_opts["theoretical_volatility"], errors="coerce") / 100.0
        )

        self.all_opts["K"] = pd.to_numeric(self.all_opts["strike"], errors="coerce")
        self.all_opts["OI"] = pd.to_numeric(self.all_opts["open_interest"], errors="coerce")

        self.all_opts = self.all_opts.dropna(subset=["iv", "K", "OI", "T"])
        self.all_opts = self.all_opts[(self.all_opts["iv"] > 0) & (self.all_opts["OI"] > 0)].copy()

        self.spot = float(
            pd.to_numeric(self.all_opts["underlying_price"], errors="coerce").dropna().iloc[0]
        )

    def plot(self):
        """
        Generate and display the GEX by price plot.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if self.all_opts is None:
            self.load_data()

        is_call = (self.all_opts["contract_type"] == "CALL").to_numpy()
        k = self.all_opts["K"].to_numpy(dtype=float)
        t = self.all_opts["T"].to_numpy(dtype=float)
        iv = self.all_opts["iv"].to_numpy(dtype=float)
        oi = self.all_opts["OI"].to_numpy(dtype=float)

        prices_grid = np.arange(round(self.spot) - 300, round(self.spot) + 301, 1)

        if self.debug:
            print(f"Calculating GEX on price grid from {prices_grid[0]} to {prices_grid[-1]}")

        net_gex_by_price = {}

        for p in prices_grid:
            s = np.full_like(k, float(p), dtype=float)

            # Calculate gamma using Black-Scholes
            gam = bs_gamma(s=s, k=k, t=t, sigma=iv, r=0.0, q=0.0)

            # GEX scaling: gamma * OI * price^2
            gex_each = gam * oi * (float(p) ** 2)

            # Net GEX = calls - puts
            net_gex = gex_each[is_call].sum() - gex_each[~is_call].sum()
            net_gex_by_price[float(p)] = float(net_gex)

        prices = np.array(sorted(net_gex_by_price.keys()), dtype=float)
        gex = np.array([net_gex_by_price[p] for p in prices], dtype=float)

        # Find zero-gamma crossing (linear interpolation)
        # Handle edge case where gex values might be exactly zero
        sign = np.sign(gex)

        # Replace zeros with the previous non-zero sign to avoid spurious crossings
        nonzero_mask = sign != 0
        if nonzero_mask.any():
            # Forward fill the sign array where zeros occur
            sign_filled = sign.copy()
            last_nonzero = None
            for i in range(len(sign)):
                if sign[i] != 0:
                    last_nonzero = sign[i]
                elif last_nonzero is not None:
                    sign_filled[i] = last_nonzero

            # Find where sign changes (excluding spurious changes from zeros)
            idx = np.where(np.diff(sign_filled) != 0)[0]
        else:
            idx = np.array([])

        if self.debug:
            print("crossings found at indices: ", idx)

        zgl = None
        if len(idx) > 0:
            # Use the first crossing
            i = idx[0]
            x1, x2 = prices[i], prices[i + 1]
            y1, y2 = gex[i], gex[i + 1]

            # Linear interpolation to find exact zero crossing
            if y2 != y1:  # Avoid division by zero
                zgl = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prices, gex, linewidth=2, label="Net GEX vs Price")
        ax.axhline(0, linestyle="--", linewidth=1, label="Zero Gamma (y=0)")

        ax.axvline(self.spot, linestyle=":", linewidth=1.5, label="Spot")

        if zgl is not None:
            ax.axvline(zgl, linestyle="--", linewidth=1.5, label=f"ZGL ≈ {zgl:.1f}")
            ax.annotate(f"ZGL ≈ {zgl:.1f}", xy=(zgl, 0), xytext=(zgl + 5, gex.max() * 0.05))

        # Calculate and annotate slope at spot
        i = np.argmin(np.abs(prices - self.spot))

        if 0 < i < len(prices) - 1:
            slope = (gex[i + 1] - gex[i - 1]) / (prices[i + 1] - prices[i - 1])
        else:
            slope = np.nan

        ax.scatter(prices[i], gex[i], zorder=5)
        ax.annotate(
            f"Slope @ spot ≈ {slope:,.2e}", xy=(prices[i], gex[i]), xytext=(prices[i] + 5, gex[i])
        )

        ax.set_xlabel(f"{self.symbol} Price")
        ax.set_ylabel("Net GEX (gamma · OI · S² units)")
        ax.set_title(f"{self.symbol} Net Gamma Exposure vs Price ({self.days_out}d window)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax
