from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class OpenInterestComparison:
    """Open Interest Comparison charting utilities.

    This class provides methods to visualize aggregate open interest by strike
    with grouped bars representing different expiration dates for side-by-side comparison.
    """

    def __init__(self, csv_path=None, dataframe=None, data_dir="data"):
        """Initialize OpenInterestComparison chart.

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
        """Plot open interest by strike with grouped bars by expiration for side-by-side comparison.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            min_strike: Minimum strike to display (optional)
            max_strike: Maximum strike to display (optional)
            top_n_strikes: Show only the N strikes with the most total open interest (optional)
            contract_type: Type of contracts to display - "ALL", "CALL", or "PUT" (default: "ALL")

        Returns:
            Tuple of (fig, ax)
        """
        if self.df is None or self.df.empty:
            raise ValueError(
                "No data to plot. Call load_data() first or provide dataframe in __init__"
            )

        df_filtered = self.df.copy()
        if contract_type in ["CALL", "PUT"]:
            df_filtered = df_filtered[df_filtered["contract_type"] == contract_type]
        elif contract_type != "ALL":
            raise ValueError(
                f"contract_type must be 'ALL', 'CALL', or 'PUT', got '{contract_type}'"
            )

        oi_by_strike_exp = (
            df_filtered.groupby(["strike", "expiration_date"])["open_interest"]
            .sum()
            .unstack(fill_value=0)
        )

        oi_by_strike_exp = oi_by_strike_exp.sort_index()

        if min_strike is not None or max_strike is not None:
            if min_strike is not None and max_strike is not None:
                oi_by_strike_exp = oi_by_strike_exp.loc[
                    (oi_by_strike_exp.index >= min_strike) & (oi_by_strike_exp.index <= max_strike)
                ]
            elif min_strike is not None:
                oi_by_strike_exp = oi_by_strike_exp.loc[oi_by_strike_exp.index >= min_strike]
            elif max_strike is not None:
                oi_by_strike_exp = oi_by_strike_exp.loc[oi_by_strike_exp.index <= max_strike]

        # NOTE: filter to top N strikes by total open interest
        if top_n_strikes is not None:
            # Calculate total OI across all expirations for each strike
            total_oi_by_strike = oi_by_strike_exp.sum(axis=1)
            # Get the top N strikes
            top_strikes = total_oi_by_strike.nlargest(top_n_strikes).index
            # Filter to only those strikes
            oi_by_strike_exp = oi_by_strike_exp.loc[top_strikes]
            # Re-sort by strike price
            oi_by_strike_exp = oi_by_strike_exp.sort_index()

        fig, ax = plt.subplots(figsize=figsize)

        oi_by_strike_exp.plot(kind="bar", stacked=False, ax=ax, width=0.8, colormap="tab20")

        contract_label = "All Contracts" if contract_type == "ALL" else f"{contract_type}s"
        ax.set_title(
            f"Open Interest Comparison by Strike and Expiration ({contract_label})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Strike Price", fontsize=12)
        ax.set_ylabel("Open Interest", fontsize=12)
        ax.legend(title="Expiration", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        ax.set_xticklabels(
            [f"{int(strike)}" for strike in oi_by_strike_exp.index], rotation=45, ha="right"
        )

        # NOTE: Reduce number of x-ticks if too many strikes
        if len(oi_by_strike_exp) > 50:
            step = len(oi_by_strike_exp) // 25
            for i, label in enumerate(ax.get_xticklabels()):
                if i % step != 0:
                    label.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def load_data(self, symbol, start_date, days_out=7):
        """Load option chain data for a specified number of days starting from the given date.

        This method loads the most recent option chain snapshot for each day
        in the specified period (start_date + days_out days).

        Args:
            symbol: Trading symbol (e.g., '$SPX', 'SPXW')
            start_date: Starting date in YYYY-MM-DD format
            days_out: Number of days to include (default: 7). If days_out=1, only start_date is used.

        Returns:
            DataFrame with aggregated option chain data
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=days_out)

        # Find all CSV files for this symbol
        months = [start_dt.month, (start_dt.month % 12) + 1]
        years = [start_dt.year, start_dt.year if start_dt.month < 12 else start_dt.year + 1]

        csv_files = []
        for m, y in zip(months, years):
            month_str = f"{y:04d}-{m:02d}"
            pattern = f"{symbol}_exp{month_str}*.csv"
            csv_files.extend(sorted(self.data_dir.glob(pattern)))

        if not csv_files:
            raise ValueError(
                f"No option chain CSV files found for symbol {symbol} in {self.data_dir}"
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
                f"No option chain files found with expirations between {start_date} and {end_dt.strftime('%Y-%m-%d')}"
            )

        latest_files = [file_info[1] for file_info in files_by_expiration.values()]

        dfs = []
        for csv_file in latest_files:
            df_temp = pd.read_csv(csv_file)

            if not df_temp.empty:
                dfs.append(df_temp)

        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("All loaded dataframes are empty")
        self._prepare_data()

        return self.df

    def _prepare_data(self):
        """Ensure numeric columns are parsed properly and aggregate by strike/expiration."""
        numeric_columns = ["strike", "open_interest"]
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
