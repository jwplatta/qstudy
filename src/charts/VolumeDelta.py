from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class VolumeDelta:
    def __init__(self, csv_path=None, dataframe=None, data_dir="data"):
        """Initialize VolumeDelta chart.

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
        figsize=(20, 8),
        save_path=None,
        min_strike=None,
        max_strike=None,
        top_n_strikes=None,
        contract_type="ALL",
    ):
        """Plot volume delta and latest volume by strike for a single expiration.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            min_strike: Minimum strike to display (optional)
            max_strike: Maximum strike to display (optional)
            top_n_strikes: Show only the N strikes with the most absolute volume delta (optional)
            contract_type: Type of contracts to display - "ALL", "CALL", or "PUT" (default: "ALL")

        Returns:
            Tuple of (fig, axes)
        """
        if self.df is None or self.df.empty:
            raise ValueError(
                "No data to plot. Call load_data() first or provide dataframe in __init__"
            )

        df_filtered = self.df.copy()

        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if contract_type == "ALL":
            call_delta = (
                df_filtered[df_filtered["contract_type"] == "CALL"]
                .groupby("strike")["volume_delta"]
                .sum()
            )
            put_delta = (
                df_filtered[df_filtered["contract_type"] == "PUT"]
                .groupby("strike")["volume_delta"]
                .sum()
            )

            call_volume = (
                df_filtered[df_filtered["contract_type"] == "CALL"]
                .groupby("strike")["total_volume_latest"]
                .sum()
            )
            put_volume = (
                df_filtered[df_filtered["contract_type"] == "PUT"]
                .groupby("strike")["total_volume_latest"]
                .sum()
            )

            # Combine into DataFrames with all strikes
            all_strikes = sorted(set(call_delta.index) | set(put_delta.index))
            delta_data = pd.DataFrame(
                {
                    "CALL": call_delta.reindex(all_strikes, fill_value=0),
                    "PUT": put_delta.reindex(all_strikes, fill_value=0),
                }
            )
            volume_data = pd.DataFrame(
                {
                    "CALL": call_volume.reindex(all_strikes, fill_value=0),
                    "PUT": put_volume.reindex(all_strikes, fill_value=0),
                }
            )

            if min_strike is not None or max_strike is not None:
                if min_strike is not None and max_strike is not None:
                    delta_data = delta_data.loc[
                        (delta_data.index >= min_strike) & (delta_data.index <= max_strike)
                    ]
                    volume_data = volume_data.loc[
                        (volume_data.index >= min_strike) & (volume_data.index <= max_strike)
                    ]
                elif min_strike is not None:
                    delta_data = delta_data.loc[delta_data.index >= min_strike]
                    volume_data = volume_data.loc[volume_data.index >= min_strike]
                elif max_strike is not None:
                    delta_data = delta_data.loc[delta_data.index <= max_strike]
                    volume_data = volume_data.loc[volume_data.index <= max_strike]

            # Filter to top N strikes by absolute volume delta
            if top_n_strikes is not None:
                total_abs_delta = delta_data.abs().sum(axis=1)
                top_strikes = total_abs_delta.nlargest(top_n_strikes).index
                delta_data = delta_data.loc[top_strikes]
                volume_data = volume_data.loc[top_strikes]
                delta_data = delta_data.sort_index()
                volume_data = volume_data.sort_index()

            # Plot volume delta
            delta_data.plot(kind="bar", ax=ax1, width=0.8, color=["green", "red"])

            # Plot latest volume
            volume_data.plot(kind="bar", ax=ax2, width=0.8, color=["green", "red"])

            strikes = delta_data.index

        else:
            if contract_type not in ["CALL", "PUT"]:
                raise ValueError(
                    f"contract_type must be 'ALL', 'CALL', or 'PUT', got '{contract_type}'"
                )

            df_filtered = df_filtered[df_filtered["contract_type"] == contract_type]
            delta_by_strike = df_filtered.groupby("strike")["volume_delta"].sum().sort_index()
            volume_by_strike = (
                df_filtered.groupby("strike")["total_volume_latest"].sum().sort_index()
            )

            if min_strike is not None or max_strike is not None:
                if min_strike is not None and max_strike is not None:
                    delta_by_strike = delta_by_strike.loc[
                        (delta_by_strike.index >= min_strike)
                        & (delta_by_strike.index <= max_strike)
                    ]
                    volume_by_strike = volume_by_strike.loc[
                        (volume_by_strike.index >= min_strike)
                        & (volume_by_strike.index <= max_strike)
                    ]
                elif min_strike is not None:
                    delta_by_strike = delta_by_strike.loc[delta_by_strike.index >= min_strike]
                    volume_by_strike = volume_by_strike.loc[volume_by_strike.index >= min_strike]
                elif max_strike is not None:
                    delta_by_strike = delta_by_strike.loc[delta_by_strike.index <= max_strike]
                    volume_by_strike = volume_by_strike.loc[volume_by_strike.index <= max_strike]

            # Filter to top N strikes by absolute volume delta
            if top_n_strikes is not None:
                top_strikes = delta_by_strike.abs().nlargest(top_n_strikes).index
                delta_by_strike = delta_by_strike.loc[top_strikes]
                volume_by_strike = volume_by_strike.loc[top_strikes]
                delta_by_strike = delta_by_strike.sort_index()
                volume_by_strike = volume_by_strike.sort_index()

            color = "green" if contract_type == "CALL" else "red"

            # Plot volume delta
            delta_by_strike.plot(kind="bar", ax=ax1, width=0.8, color=color)

            # Plot latest volume
            volume_by_strike.plot(kind="bar", ax=ax2, width=0.8, color=color)

            strikes = delta_by_strike.index

        # Add vertical line at spot price for both subplots
        if "underlying_price" in self.df.columns:
            spot_price = self.df["underlying_price"].iloc[0]
            strike_positions = {strike: i for i, strike in enumerate(strikes)}
            closest_strike = min(strikes, key=lambda x: abs(x - spot_price))
            spot_position = strike_positions[closest_strike]

            for ax in [ax1, ax2]:
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

        # Configure left subplot (Volume Delta)
        ax1.set_title(
            f"Volume Delta by Strike - {expiration} ({contract_label})",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Strike Price", fontsize=12)
        ax1.set_ylabel("Volume Delta", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Configure right subplot (Latest Volume)
        ax2.set_title(
            f"Latest Volume by Strike - {expiration} ({contract_label})",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Strike Price", fontsize=12)
        ax2.set_ylabel("Volume", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Sync y-axis ranges for both subplots
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        # Use the same range for both (take the max range)
        global_min = min(y1_min, y2_min)
        global_max = max(y1_max, y2_max)

        ax1.set_ylim(global_min, global_max)
        ax2.set_ylim(global_min, global_max)

        # Set x-tick labels for both subplots
        for ax in [ax1, ax2]:
            ax.set_xticklabels([f"{int(strike)}" for strike in strikes], rotation=45, ha="right")

            # Show ticks at multiples of 10 strikes
            for i, (label, strike) in enumerate(zip(ax.get_xticklabels(), strikes)):
                if int(strike) % 10 != 0:
                    label.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, (ax1, ax2)

    def load_data(self, symbol=None, expiration_date=None, lookback=1):
        """Load option chain snapshots and calculate volume delta.

        This method finds option chain snapshots for the given expiration date
        and calculates the volume delta between the latest snapshot and a previous
        snapshot based on the lookback parameter.

        Args:
            symbol: Trading symbol (e.g., 'SPXW', '$SPX')
            expiration_date: Expiration date in YYYY-MM-DD format
            lookback: Number of snapshots to look back from the latest (default: 1)
                     lookback=1 means latest - second_latest
                     lookback=2 means latest - third_latest, etc.

        Returns:
            DataFrame with volume delta data
        """
        pattern = f"{symbol}_exp{expiration_date}_*.csv"
        csv_files = sorted(self.data_dir.glob(pattern))

        if not csv_files:
            raise ValueError(
                f"No option chain CSV files found for {symbol} with expiration {expiration_date} in {self.data_dir}"
            )

        if len(csv_files) < lookback + 1:
            raise ValueError(
                f"Need at least {lookback + 1} snapshots to calculate volume delta with lookback={lookback}, "
                f"found only {len(csv_files)}"
            )

        # Get the latest snapshot and the one at lookback position
        latest_file = csv_files[-1]
        previous_file = csv_files[-(lookback + 1)]
        print("Latest snapshot file:", latest_file)
        print("Previous snapshot file:", previous_file)

        # Load both snapshots
        latest_df = pd.read_csv(latest_file)
        previous_df = pd.read_csv(previous_file)

        if latest_df.empty or previous_df.empty:
            raise ValueError("One or both snapshot files are empty")

        # Calculate volume delta for each contract
        # Merge on contract identifier (strike + contract_type)
        latest_df = latest_df[
            ["contract_type", "strike", "total_volume", "expiration_date", "underlying_price"]
        ]
        previous_df = previous_df[["contract_type", "strike", "total_volume"]]

        merged = latest_df.merge(
            previous_df,
            on=["contract_type", "strike"],
            how="outer",
            suffixes=("_latest", "_previous"),
        )

        # Fill NaN with 0 (for new contracts that didn't exist in previous snapshot)
        # merged["total_volume_latest"] = merged["total_volume_latest"].fillna(0)
        # merged["total_volume_previous"] = merged["total_volume_previous"].fillna(0)

        # Calculate delta
        merged["volume_delta"] = merged["total_volume_latest"] - merged["total_volume_previous"]

        self.df = merged
        self._prepare_data()

        return self.df

    def _prepare_data(self):
        """Ensure numeric columns are parsed properly."""
        numeric_columns = ["strike", "volume_delta", "underlying_price"]
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
