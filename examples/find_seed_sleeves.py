"""
Identify candidate seed sleeves for walkforward_greedy_portfolio.py.

Loads all signal-sweep yearly results, filters to pool-candidate sleeves,
then ranks by consistency: good average net Sharpe, few negative years,
no extreme drawdowns, reasonable turnover.

Outputs:
  examples/out/seed_sleeve_candidates.csv  — ranked table
  examples/out/seed_sleeve_candidates.png  — scatter of avg vs min annual SR
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).parent / "out"
SWEEP_OUT = Path(__file__).parent / "signal_sweeps" / "out"

# ---------------------------------------------------------------------------
# Thresholds — tune these to taste
# ---------------------------------------------------------------------------
MIN_YEARS_COVERED = 7          # must have data for at least this many years
MIN_AVG_NET_SHARPE = 0.15      # average annual net Sharpe across all years
MIN_ANNUAL_NET_SHARPE = -0.50  # worst single year allowed
MAX_PCT_NEGATIVE_YEARS = 0.45  # at most this fraction of years can be negative
MAX_AVG_DRAWDOWN = -0.25       # average annual max drawdown (less negative = better)
MAX_WORST_DRAWDOWN = -0.50     # single-year worst drawdown allowed
MAX_AVG_TURNOVER = 2.0         # average daily turnover cap (round-trips)


def load_yearly() -> pd.DataFrame:
    dfs = []
    for f in SWEEP_OUT.glob("*/signal_sweep_*_yearly.csv"):
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)


def load_pool_candidates() -> set[str]:
    dfs = []
    for f in SWEEP_OUT.glob("*/signal_sweep_*_pool_candidates.csv"):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    return set(combined["name"].unique())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yearly = load_yearly()
    pool_names = load_pool_candidates()

    print(f"Total sleeves in yearly data : {yearly['name'].nunique()}")
    print(f"Pool-candidate sleeves       : {len(pool_names)}")

    # Restrict to pool candidates
    yearly = yearly[yearly["name"].isin(pool_names)].copy()
    print(f"Rows after pool-candidate filter: {len(yearly)}")

    # ---------------------------------------------------------------------------
    # Per-sleeve aggregate stats
    # ---------------------------------------------------------------------------
    grp = yearly.groupby("name")

    agg = pd.DataFrame(
        {
            "n_years": grp["year"].count(),
            "avg_net_sharpe": grp["net_sharpe"].mean(),
            "min_net_sharpe": grp["net_sharpe"].min(),
            "max_net_sharpe": grp["net_sharpe"].max(),
            "std_net_sharpe": grp["net_sharpe"].std(),
            "pct_negative_years": grp["net_sharpe"].apply(lambda s: (s < 0).mean()),
            "avg_max_drawdown": grp["max_drawdown"].mean(),
            "worst_max_drawdown": grp["max_drawdown"].min(),
            "avg_turnover": grp["avg_daily_turnover"].mean(),
        }
    ).reset_index()

    # Consistency score: information ratio of annual returns (mean/std)
    agg["consistency_score"] = agg["avg_net_sharpe"] / agg["std_net_sharpe"].replace(0, float("nan"))

    print(f"\nBefore filters: {len(agg)} sleeves")

    # ---------------------------------------------------------------------------
    # Apply filters
    # ---------------------------------------------------------------------------
    mask = (
        (agg["n_years"] >= MIN_YEARS_COVERED)
        & (agg["avg_net_sharpe"] >= MIN_AVG_NET_SHARPE)
        & (agg["min_net_sharpe"] >= MIN_ANNUAL_NET_SHARPE)
        & (agg["pct_negative_years"] <= MAX_PCT_NEGATIVE_YEARS)
        & (agg["avg_max_drawdown"] >= MAX_AVG_DRAWDOWN)
        & (agg["worst_max_drawdown"] >= MAX_WORST_DRAWDOWN)
        & (agg["avg_turnover"] <= MAX_AVG_TURNOVER)
    )
    candidates = agg[mask].copy()
    print(f"After filters : {len(candidates)} sleeves")

    # ---------------------------------------------------------------------------
    # Rank: primary = consistency_score, secondary = avg_net_sharpe
    # ---------------------------------------------------------------------------
    candidates = candidates.sort_values(
        ["consistency_score", "avg_net_sharpe"], ascending=False
    ).reset_index(drop=True)
    candidates.index += 1  # 1-based rank
    candidates.index.name = "rank"

    # Save
    out_csv = OUT_DIR / "seed_sleeve_candidates.csv"
    candidates.to_csv(out_csv)
    print(f"\nSaved {out_csv}")
    print(candidates[["name", "avg_net_sharpe", "min_net_sharpe", "pct_negative_years",
                       "avg_max_drawdown", "avg_turnover", "consistency_score"]].head(20).to_string())

    # ---------------------------------------------------------------------------
    # Chart: avg vs min annual Sharpe, coloured by pct_negative_years
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        candidates["avg_net_sharpe"],
        candidates["min_net_sharpe"],
        c=candidates["pct_negative_years"],
        cmap="RdYlGn_r",
        s=60,
        alpha=0.8,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Fraction of negative years")

    # Label top 10
    for _, row in candidates.head(10).iterrows():
        ax.annotate(
            row["name"],
            xy=(row["avg_net_sharpe"], row["min_net_sharpe"]),
            fontsize=6,
            xytext=(4, 2),
            textcoords="offset points",
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average annual net Sharpe")
    ax.set_ylabel("Worst single-year net Sharpe")
    ax.set_title("Seed sleeve candidates — consistency view\n(top 10 labelled)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_png = OUT_DIR / "seed_sleeve_candidates.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")

    # ---------------------------------------------------------------------------
    # Print the SEED_SLEEVE list ready to paste into walkforward_greedy_portfolio.py
    # ---------------------------------------------------------------------------
    print("\n# Top 10 seed sleeve candidates — paste into SEED_SLEEVE:")
    for rank, row in candidates.head(10).iterrows():
        print(
            f'  #{rank:2d}  avg_SR={row["avg_net_sharpe"]:.2f}  '
            f'min_SR={row["min_net_sharpe"]:.2f}  '
            f'neg_yrs={row["pct_negative_years"]:.0%}  '
            f'"{row["name"]}"'
        )


if __name__ == "__main__":
    main()
