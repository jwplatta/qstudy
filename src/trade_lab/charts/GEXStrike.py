from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


class GEXStrike:
    MULTIPLIER = 100

    def __init__(self, symbol=None, start_date=None, days_out=10, data_dir="data", debug=False):
        """
        Initialize GEXStrike chart.

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
            print(f"Loading {len(latest_files)} option chain files for GEX by Strike calculation.")
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
        Generate and display the GEX by strike plot.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if self.all_opts is None:
            self.load_data()

        is_call = (self.all_opts["contract_type"] == "CALL").to_numpy()
        k = self.all_opts["K"].to_numpy(dtype=float)
        oi = self.all_opts["OI"].to_numpy(dtype=float)
        gam = pd.to_numeric(self.all_opts["gamma"], errors="coerce").to_numpy(dtype=float)

        gex_each = gam * oi * (self.spot**2)

        gex_df = pd.DataFrame({"strike": k, "is_call": is_call, "gex": gex_each})

        net = (
            gex_df.assign(sign=np.where(gex_df["is_call"], 1.0, -1.0))
            .assign(net_gex=lambda d: d["gex"] * d["sign"])
            .groupby("strike", as_index=False)["net_gex"]
            .sum()
        )

        if self.debug:
            print(f"Calculated GEX for {len(net)} unique strikes")

        strikes = net["strike"].to_numpy(float)
        gex = net["net_gex"].to_numpy(float)

        strike_range = 300
        mask = (strikes >= self.spot - strike_range) & (strikes <= self.spot + strike_range)
        strikes = strikes[mask]
        gex = gex[mask]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(strikes, gex, width=5.0, label="Net GEX by Strike", alpha=0.7)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(
            self.spot, color="red", linestyle=":", linewidth=2, label=f"Spot = {self.spot:.1f}"
        )

        # NOTE: Highlight strikes with largest absolute GEX
        abs_gex = np.abs(gex)
        if len(abs_gex) > 0:
            top_n = min(3, len(abs_gex))
            top_indices = np.argsort(abs_gex)[-top_n:]
            for idx in top_indices:
                ax.annotate(
                    f"{strikes[idx]:.0f}\n{gex[idx]:,.0f}",
                    xy=(strikes[idx], gex[idx]),
                    xytext=(0, 10 if gex[idx] > 0 else -20),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                )

        ax.xaxis.set_major_locator(MultipleLocator(25))

        ax.set_xlabel(f"{self.symbol} Strike Price")
        ax.set_ylabel("Net GEX (gamma · OI · S² units)")
        ax.set_title(f"{self.symbol} Net Gamma Exposure by Strike ({self.days_out}d window)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig, ax
