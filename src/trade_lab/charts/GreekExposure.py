from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


class GreekExposure:
    """Chart class for visualizing exposure for multiple greeks across strike prices."""

    VALID_GREEKS = ("gamma", "vanna", "charm")

    def __init__(
        self,
        symbol=None,
        start_date=None,
        days_out=10,
        data_dir="data",
        greek="gamma",
        multiplier=100,
        dealer_sign=1.0,
        debug=False,
    ):
        """
        Initialize GreekExposure chart.

        Args:
            symbol: The ticker symbol (e.g., 'SPXW', 'SPX')
            start_date: Start date as string 'YYYY-MM-DD' (defaults to today)
            days_out: Number of calendar days to include expirations (default: 10)
            data_dir: Directory containing option chain CSV files
            greek: One of 'gamma', 'vanna', 'charm' (default: 'gamma')
            multiplier: Contract multiplier (default: 100)
            dealer_sign: Sign for dealer positioning (default: 1.0 for raw exposure,
                         use -1.0 if assuming dealers are short customer OI)
            debug: Verbose output flag
        """
        if days_out > 45:
            raise ValueError("days_out should not exceed 45 days to limit data volume.")

        if greek not in self.VALID_GREEKS:
            raise ValueError(f"greek must be one of {self.VALID_GREEKS}, got '{greek}'")

        self.symbol = symbol
        self.start_date = start_date or datetime.now().strftime("%Y-%m-%d")
        self.days_out = days_out
        self.data_dir = Path(data_dir)
        self.greek = greek
        self.multiplier = multiplier
        self.dealer_sign = dealer_sign
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
                f"No option chain files found with expirations between "
                f"{self.start_date} and {end_dt.strftime('%Y-%m-%d')}"
            )

        latest_files = [file_info[1] for file_info in files_by_expiration.values()]
        self.asof = min(file_info[0] for file_info in files_by_expiration.values())

        if self.debug:
            print(
                f"Loading {len(latest_files)} option chain files for "
                f"{self.greek} exposure calculation."
            )
            print(f"Data as of {self.asof.strftime('%Y-%m-%d %H:%M:%S')}")

        dfs = []
        for csv_file in latest_files:
            if self.debug:
                print(csv_file)

            df_temp = pd.read_csv(csv_file)
            if not df_temp.empty:
                dfs.append(df_temp)

        self.all_opts = pd.concat(dfs, ignore_index=True).copy()

        # Convert expiration date to datetime when trading stops (3 PM CT expiry)
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

        # Ensure we have required columns for the selected greek
        required_cols = ["iv", "K", "OI", "T"]
        if self.greek == "vanna" and "vega" in self.all_opts.columns:
            self.all_opts["vega"] = pd.to_numeric(self.all_opts["vega"], errors="coerce")
            required_cols.append("vega")
        if self.greek == "charm" and "theta" in self.all_opts.columns:
            self.all_opts["theta"] = pd.to_numeric(self.all_opts["theta"], errors="coerce")
            required_cols.append("theta")
        if self.greek == "gamma":
            self.all_opts["gamma"] = pd.to_numeric(self.all_opts["gamma"], errors="coerce")
            required_cols.append("gamma")

        self.all_opts = self.all_opts.dropna(subset=required_cols)
        self.all_opts = self.all_opts[(self.all_opts["iv"] > 0) & (self.all_opts["OI"] > 0)].copy()

        self.spot = float(
            pd.to_numeric(self.all_opts["underlying_price"], errors="coerce").dropna().iloc[0]
        )

    def _calc_vanna(self, df):
        """
        Calculate vanna per expiration and contract type using np.gradient(vega, strike).

        Groups by both expiration_date and contract_type to ensure unique strikes
        within each group, avoiding divide-by-zero errors in gradient calculation.

        Args:
            df: DataFrame with vega and strike columns

        Returns:
            DataFrame with vanna column added
        """
        result = []
        for (exp, contract_type), group in df.groupby(["expiration_date", "contract_type"]):
            g = group.sort_values("K").copy()
            strikes = g["K"].to_numpy(float)
            vega = g["vega"].to_numpy(float)
            if len(strikes) > 1:
                g["vanna"] = np.gradient(vega, strikes)
            else:
                g["vanna"] = 0.0
            result.append(g)
        return pd.concat(result, ignore_index=True)

    def _calc_charm(self, df):
        """
        Calculate charm per expiration and contract type using np.gradient(theta, strike).

        Groups by both expiration_date and contract_type to ensure unique strikes
        within each group, avoiding divide-by-zero errors in gradient calculation.

        Args:
            df: DataFrame with theta and strike columns

        Returns:
            DataFrame with charm column added
        """
        result = []
        for (exp, contract_type), group in df.groupby(["expiration_date", "contract_type"]):
            g = group.sort_values("K").copy()
            strikes = g["K"].to_numpy(float)
            theta = g["theta"].to_numpy(float)
            if len(strikes) > 1:
                g["charm"] = np.gradient(theta, strikes)
            else:
                g["charm"] = 0.0
            result.append(g)
        return pd.concat(result, ignore_index=True)

    def _get_greek_values(self):
        """
        Get the greek values based on the selected greek type.

        Returns:
            numpy array of greek values
        """
        if self.greek == "gamma":
            return self.all_opts["gamma"].to_numpy(dtype=float)
        elif self.greek == "vanna":
            if "vega" not in self.all_opts.columns:
                raise ValueError("vega column required for vanna calculation")
            self.all_opts = self._calc_vanna(self.all_opts)
            return self.all_opts["vanna"].to_numpy(dtype=float)
        elif self.greek == "charm":
            if "theta" not in self.all_opts.columns:
                raise ValueError("theta column required for charm calculation")
            self.all_opts = self._calc_charm(self.all_opts)
            return self.all_opts["charm"].to_numpy(dtype=float)

    def plot(self, figsize=(14, 7), save_path=None):
        """
        Generate and display the greek exposure by strike plot.

        Args:
            figsize: Figure size (width, height) in inches (default: (14, 7))
            save_path: Optional path to save the figure (default: None)
                       Pass True to save to {symbol}_{greek}_exposure.png

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        if self.all_opts is None:
            self.load_data()

        is_call = (self.all_opts["contract_type"] == "CALL").to_numpy()
        k = self.all_opts["K"].to_numpy(dtype=float)
        oi = self.all_opts["OI"].to_numpy(dtype=float)
        greek_values = self._get_greek_values()

        # Exposure formula: greek_value * open_interest * multiplier * spot
        exposure = greek_values * oi * self.multiplier * self.spot

        exposure_df = pd.DataFrame({"strike": k, "is_call": is_call, "exposure": exposure})

        # Aggregate across expirations
        # For gamma: apply call/put sign convention (calls +1, puts -1)
        # For vanna/charm: no call/put sign flip (the greek itself encodes directionality)
        if self.greek == "gamma":
            net = (
                exposure_df.assign(sign=np.where(exposure_df["is_call"], 1.0, -1.0))
                .assign(net_exposure=lambda d: d["exposure"] * d["sign"])
                .groupby("strike", as_index=False)["net_exposure"]
                .sum()
            )
        else:
            # Vanna/Charm: sum raw exposure without call/put sign flip
            net = exposure_df.groupby("strike", as_index=False)["exposure"].sum()
            net = net.rename(columns={"exposure": "net_exposure"})

        # Apply dealer sign (use -1.0 if assuming dealers are short customer OI)
        net["net_exposure"] = net["net_exposure"] * self.dealer_sign

        if self.debug:
            print(f"Calculated {self.greek} exposure for {len(net)} unique strikes")

        strikes = net["strike"].to_numpy(float)
        exposure_vals = net["net_exposure"].to_numpy(float)

        # Filter to strike range: +/- 300 points from spot
        strike_range = 300
        mask = (strikes >= self.spot - strike_range) & (strikes <= self.spot + strike_range)
        strikes = strikes[mask]
        exposure_vals = exposure_vals[mask]

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(
            strikes,
            exposure_vals,
            width=5.0,
            label=f"Net {self.greek.capitalize()} Exposure by Strike",
            alpha=0.7,
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(
            self.spot, color="red", linestyle=":", linewidth=2, label=f"Spot = {self.spot:.1f}"
        )

        # Highlight strikes with largest absolute exposure
        abs_exposure = np.abs(exposure_vals)
        if len(abs_exposure) > 0:
            top_n = min(3, len(abs_exposure))
            top_indices = np.argsort(abs_exposure)[-top_n:]
            for idx in top_indices:
                ax.annotate(
                    f"{strikes[idx]:.0f}\n{exposure_vals[idx]:,.0f}",
                    xy=(strikes[idx], exposure_vals[idx]),
                    xytext=(0, 10 if exposure_vals[idx] > 0 else -20),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                )

        ax.xaxis.set_major_locator(MultipleLocator(25))

        greek_title = self.greek.capitalize()
        ax.set_xlabel(f"{self.symbol} Strike Price")
        ax.set_ylabel(f"Net {greek_title} Exposure")
        ax.set_title(
            f"{self.symbol} Net {greek_title} Exposure by Strike ({self.days_out}d window)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            if save_path is True:
                save_path = f"{self.symbol}_{self.greek}_exposure.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
