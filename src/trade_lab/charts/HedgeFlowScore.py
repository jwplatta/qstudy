from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class HedgeFlowScore:
    def __init__(self, data_dir="data", spot_window_pct=0.01, reference_move_pct=0.0025):
        """
        Initialize HedgeFlowScore calculator and plotter.

        Args:
            data_dir: Directory containing option chain CSV files
            spot_window_pct: Window around spot for local gamma (default 1% = ±0.5%)
            reference_move_pct: Reference move for HFS calculation (default 0.25%)
        """
        self.data_dir = Path(data_dir)
        self.spot_window_pct = spot_window_pct
        self.reference_move_pct = reference_move_pct
        self.timestamps = []
        self.hfs_scores = []

    def plot(self, figsize=(14, 7), save_path=None):
        """
        Plot HedgeFlowScore over time with regime threshold lines.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if not self.timestamps:
            raise ValueError("No data to plot. Call load_and_calculate() first.")

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.timestamps, self.hfs_scores, "b-", linewidth=2, label="HFS Score")
        ax.scatter(self.timestamps, self.hfs_scores, c="blue", s=30, alpha=0.6, zorder=5)

        # Step 5: Add regime threshold lines
        ax.axhline(
            y=0.25,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="Mean Reversion / Pinning (+0.25)",
            alpha=0.7,
        )
        ax.axhline(
            y=-0.25,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Acceleration / Breakout (-0.25)",
            alpha=0.7,
        )
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.fill_between(
            self.timestamps, 0.25, 1.0, alpha=0.1, color="green", label="Reversion Zone"
        )
        ax.fill_between(self.timestamps, -0.25, -1.0, alpha=0.1, color="red", label="Breakout Zone")
        ax.fill_between(
            self.timestamps, -0.25, 0.25, alpha=0.05, color="yellow", label="Fragile / Chop Zone"
        )

        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("HFS Normalized Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Hedge Flow Score - Intraday Decision Metric", fontsize=14, fontweight="bold", pad=20
        )
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        fig.autofmt_xdate()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Chart saved to {save_path}")

        return fig, ax

    def load_and_calculate(self, expiration_filter=None):
        """Load all option chain CSV files and calculate HFS for each timestamp.

        Args:
            expiration_filter: Expiration date string (YYYY-MM-DD) to filter files (required)
        """
        if expiration_filter is None:
            raise ValueError("expiration_filter is required and cannot be None")

        # Filter by expiration date: SPXW_exp2025-12-24_*.csv
        pattern = f"*_exp{expiration_filter}_*.csv"

        csv_files = sorted(self.data_dir.glob(pattern))

        if not csv_files:
            filter_msg = f" for expiration {expiration_filter}" if expiration_filter else ""
            raise ValueError(f"No option chain CSV files found in {self.data_dir}{filter_msg}")

        for csv_file in csv_files:
            try:
                # Parse timestamp from filename: $SPX_exp2025-12-24_2025-12-18_14-30-00.csv
                # Format: {symbol}_exp{expiration_date}_{fetch_date}_{fetch_time}.csv
                parts = csv_file.stem.split("_")
                if len(parts) >= 4:
                    # Parts: ['$SPX', 'exp2025-12-24', '2025-12-18', '14-30-00']
                    fetch_date = parts[2]
                    fetch_time = parts[3]
                    timestamp = datetime.strptime(f"{fetch_date}_{fetch_time}", "%Y-%m-%d_%H-%M-%S")
                else:
                    continue

                df = pd.read_csv(csv_file)

                # Calculate HFS for this snapshot
                hfs_norm = self._calculate_hfs(df)

                self.timestamps.append(timestamp)
                self.hfs_scores.append(hfs_norm)

            except Exception as e:
                print(f"Warning: Error processing {csv_file.name}: {e}")
                continue

        if not self.timestamps:
            raise ValueError("No valid option chain data with timestamps found")

    def _calculate_hfs(self, df):
        """
        Calculate normalized Hedge Flow Score following the 5-step procedure.

        Steps:
        1. Select local strike window around spot
        2. Compute signed local dealer gamma
        3. Translate gamma into expected hedge flow
        4. Normalize to obtain bounded decision score
        5. (Mapping to regime is done in plot with threshold lines)
        """
        if df.empty or "underlying_price" not in df.columns:
            return 0.0

        # Get underlying price (spot)
        spot = df["underlying_price"].iloc[0]

        # Step 1: Select local strike window (±0.5% to ±1.0% around spot)
        window_half = self.spot_window_pct / 2
        lower_bound = spot * (1 - window_half)
        upper_bound = spot * (1 + window_half)

        local_df = df[(df["strike"] >= lower_bound) & (df["strike"] <= upper_bound)].copy()

        if local_df.empty:
            return 0.0

        # Weight near-dated expiries more heavily (0-1 DTE)
        # For now, we'll use all strikes in the window equally
        # Future enhancement: add expiry-based weighting

        # Step 2: Compute signed local dealer gamma
        # For SPX, dealers are typically SHORT options (sell to customers)
        local_df["gamma_exposure"] = local_df["open_interest"] * local_df["gamma"] * 100 * spot

        call_gamma = local_df[local_df["contract_type"] == "CALL"]["gamma_exposure"].sum()
        put_gamma = local_df[local_df["contract_type"] == "PUT"]["gamma_exposure"].sum()

        # Net dealer gamma (dealers short options, so use -1)
        gamma_local = -1 * (call_gamma - put_gamma)

        # Step 3: Translate gamma into expected hedge flow
        delta_s = spot * self.reference_move_pct
        hfs = gamma_local * delta_s

        # Step 4: Normalize to obtain bounded decision score
        # The denominator represents total gamma exposure (unsigned)
        total_gamma = local_df["gamma_exposure"].abs().sum()

        if total_gamma == 0:
            return 0.0

        hfs_norm = hfs / (total_gamma * delta_s)

        # Clamp to [-1, +1] range
        hfs_norm = np.clip(hfs_norm, -1.0, 1.0)

        return hfs_norm
