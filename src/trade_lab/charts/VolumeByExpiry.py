from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class VolumeByExpiry:
    def __init__(self, csv_path=None, dataframe=None, data_dir="data"):
        """Initialize VolumeByExpiry chart.

        Args:
            csv_path: Path to CSV file containing option chain data (deprecated, use dataframe)
            dataframe: Pandas DataFrame with option chain data
            data_dir: Directory containing option chain CSV files
        """
        self.data_dir = Path(data_dir)
        self.df = None

        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
            self._prepare_data()
        elif dataframe is not None:
            self.df = dataframe.copy()
            self._prepare_data()

    def plot(
        self,
        figsize=(14, 8),
        save_path=None,
        min_strike=None,
        max_strike=None,
        top_n_strikes=None,
        contract_type="ALL",
    ):
        """Plot volume by strike for a single expiration.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            min_strike: Minimum strike to display (optional)
            max_strike: Maximum strike to display (optional)
            top_n_strikes: Show only the N strikes with the most total volume (optional)
            contract_type: Type of contracts to display - "ALL", "CALL", or "PUT" (default: "ALL")

        Returns:
            Tuple of (fig, ax)
        """
        if self.df is None or self.df.empty:
            raise ValueError(
                "No data to plot. Call load_data() first or provide dataframe in __init__"
            )

        df_filtered = self.df.copy()

        if contract_type == "ALL":
            call_volume = (
                df_filtered[df_filtered["contract_type"] == "CALL"]
                .groupby("strike")["total_volume"]
                .sum()
            )
            put_volume = (
                df_filtered[df_filtered["contract_type"] == "PUT"]
                .groupby("strike")["total_volume"]
                .sum()
            )

            # Combine into a single DataFrame with all strikes
            all_strikes = sorted(set(call_volume.index) | set(put_volume.index))
            volume_data = pd.DataFrame(
                {
                    "CALL": call_volume.reindex(all_strikes, fill_value=0),
                    "PUT": put_volume.reindex(all_strikes, fill_value=0),
                }
            )

            if min_strike is not None or max_strike is not None:
                if min_strike is not None and max_strike is not None:
                    volume_data = volume_data.loc[
                        (volume_data.index >= min_strike) & (volume_data.index <= max_strike)
                    ]
                elif min_strike is not None:
                    volume_data = volume_data.loc[volume_data.index >= min_strike]
                elif max_strike is not None:
                    volume_data = volume_data.loc[volume_data.index <= max_strike]

            # NOTE: filter to top N strikes by total volume
            if top_n_strikes is not None:
                total_volume = volume_data.sum(axis=1)
                top_strikes = total_volume.nlargest(top_n_strikes).index
                volume_data = volume_data.loc[top_strikes]
                volume_data = volume_data.sort_index()

            fig, ax = plt.subplots(figsize=figsize)
            volume_data.plot(kind="bar", ax=ax, width=0.8, color=["green", "red"])

        else:
            if contract_type not in ["CALL", "PUT"]:
                raise ValueError(
                    f"contract_type must be 'ALL', 'CALL', or 'PUT', got '{contract_type}'"
                )

            df_filtered = df_filtered[df_filtered["contract_type"] == contract_type]
            volume_by_strike = df_filtered.groupby("strike")["total_volume"].sum().sort_index()

            if min_strike is not None or max_strike is not None:
                if min_strike is not None and max_strike is not None:
                    volume_by_strike = volume_by_strike.loc[
                        (volume_by_strike.index >= min_strike)
                        & (volume_by_strike.index <= max_strike)
                    ]
                elif min_strike is not None:
                    volume_by_strike = volume_by_strike.loc[volume_by_strike.index >= min_strike]
                elif max_strike is not None:
                    volume_by_strike = volume_by_strike.loc[volume_by_strike.index <= max_strike]

            # NOTE: filter to top N strikes by total volume
            if top_n_strikes is not None:
                top_strikes = volume_by_strike.nlargest(top_n_strikes).index
                volume_by_strike = volume_by_strike.loc[top_strikes]
                volume_by_strike = volume_by_strike.sort_index()

            fig, ax = plt.subplots(figsize=figsize)
            color = "green" if contract_type == "CALL" else "red"
            volume_by_strike.plot(kind="bar", ax=ax, width=0.8, color=color)

        # Add vertical line at spot price
        if "underlying_price" in self.df.columns:
            spot_price = self.df["underlying_price"].iloc[0]
            # Get the strike index to find position
            if contract_type == "ALL":
                strikes = volume_data.index
            else:
                strikes = volume_by_strike.index

            strike_positions = {strike: i for i, strike in enumerate(strikes)}
            closest_strike = min(strikes, key=lambda x: abs(x - spot_price))
            spot_position = strike_positions[closest_strike]

            ax.axvline(
                x=spot_position,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Spot: {spot_price:.2f}",
            )

        expiration = (
            self.df["expiration_date"].iloc[0]
            if "expiration_date" in self.df.columns
            else "Unknown"
        )
        contract_label = "All Contracts" if contract_type == "ALL" else f"{contract_type}s"
        ax.set_title(
            f"Volume by Strike - {expiration} ({contract_label})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Strike Price", fontsize=12)
        ax.set_ylabel("Volume", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        if contract_type == "ALL":
            strikes = volume_data.index
        else:
            strikes = volume_by_strike.index

        ax.set_xticklabels([f"{int(strike)}" for strike in strikes], rotation=45, ha="right")

        # NOTE: Reduce number of x-ticks if too many strikes
        if len(strikes) > 50:
            step = len(strikes) // 25
            for i, label in enumerate(ax.get_xticklabels()):
                if i % step != 0:
                    label.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def load_data(self, symbol=None, expiration_date=None):
        """Load option chain data for a specific expiration date.

        This method loads the most recent option chain snapshot for the given
        expiration date.

        Args:
            symbol: Trading symbol (e.g., 'SPXW', '$SPX')
            expiration_date: Expiration date in YYYY-MM-DD format

        Returns:
            DataFrame with option chain data
        """
        pattern = f"{symbol}_exp{expiration_date}_*.csv"
        csv_files = sorted(self.data_dir.glob(pattern))

        if not csv_files:
            raise ValueError(
                f"No option chain CSV files found for {symbol} with expiration {expiration_date} in {self.data_dir}"
            )

        latest_file = csv_files[-1]

        self.df = pd.read_csv(latest_file)
        if self.df.empty:
            raise ValueError(f"Loaded dataframe from {latest_file} is empty")

        self._prepare_data()

        return self.df

    def _prepare_data(self):
        """Ensure numeric columns are parsed properly and aggregate by strike/expiration."""
        numeric_columns = ["strike", "total_volume", "underlying_price"]
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
