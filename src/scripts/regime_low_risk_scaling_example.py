import matplotlib.pyplot as plt
import pandas as pd


INITIAL_CAPITAL = 50_000
LOW_RISK_THRESH = 0.10
LOW_RISK_MULTIPLIER = 2.0


def apply_low_risk_scaling(
    trade_features: pd.DataFrame,
    low_risk_thresh: float = LOW_RISK_THRESH,
    low_risk_multiplier: float = LOW_RISK_MULTIPLIER,
) -> pd.DataFrame:
    scaled = trade_features.copy()

    required = ["regime_prob", "pnl", "total_pnl"]
    scaled = scaled.dropna(subset=required).sort_values("entry_time").reset_index(drop=True)

    scaled["is_low_risk_day"] = scaled["regime_prob"] <= low_risk_thresh
    scaled["position_multiplier"] = 1.0
    scaled.loc[scaled["is_low_risk_day"], "position_multiplier"] = low_risk_multiplier

    scaled["scaled_pnl"] = scaled["pnl"] * scaled["position_multiplier"]
    scaled["scaled_total_pnl"] = scaled["total_pnl"] * scaled["position_multiplier"]

    return scaled


def add_equity_and_drawdown(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    curve = df.copy().sort_values("entry_time").reset_index(drop=True)

    curve["equity_pnl"] = initial_capital + curve["pnl"].cumsum()
    curve["equity_total_pnl"] = initial_capital + curve["total_pnl"].cumsum()
    curve["equity_scaled_pnl"] = initial_capital + curve["scaled_pnl"].cumsum()
    curve["equity_scaled_total_pnl"] = initial_capital + curve["scaled_total_pnl"].cumsum()

    curve["peak_pnl"] = curve["equity_pnl"].cummax()
    curve["peak_total_pnl"] = curve["equity_total_pnl"].cummax()
    curve["peak_scaled_pnl"] = curve["equity_scaled_pnl"].cummax()
    curve["peak_scaled_total_pnl"] = curve["equity_scaled_total_pnl"].cummax()

    curve["drawdown_pnl"] = curve["equity_pnl"] / curve["peak_pnl"] - 1.0
    curve["drawdown_total_pnl"] = curve["equity_total_pnl"] / curve["peak_total_pnl"] - 1.0
    curve["drawdown_scaled_pnl"] = curve["equity_scaled_pnl"] / curve["peak_scaled_pnl"] - 1.0
    curve["drawdown_scaled_total_pnl"] = (
        curve["equity_scaled_total_pnl"] / curve["peak_scaled_total_pnl"] - 1.0
    )

    return curve


def summarize_low_risk_scaling(
    scaled_trades: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> None:
    low_risk_days = int(scaled_trades["is_low_risk_day"].sum())
    print("Trades:", len(scaled_trades))
    print("Low-risk scaled trades:", low_risk_days)
    print()

    print("Baseline final equity (PnL):", initial_capital + scaled_trades["pnl"].sum())
    print(
        "Baseline final equity (Total PnL):",
        initial_capital + scaled_trades["total_pnl"].sum(),
    )
    print("Scaled final equity (PnL):", initial_capital + scaled_trades["scaled_pnl"].sum())
    print(
        "Scaled final equity (Total PnL):",
        initial_capital + scaled_trades["scaled_total_pnl"].sum(),
    )


def plot_low_risk_scaling(
    scaled_trades: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    curve = add_equity_and_drawdown(scaled_trades, initial_capital=initial_capital)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_eq, ax_dd = axes

    ax_eq.plot(
        curve["entry_time"],
        curve["equity_pnl"],
        label="Baseline Equity (PnL)",
        linestyle="--",
        alpha=0.5,
        color="tab:blue",
    )
    ax_eq.plot(
        curve["entry_time"],
        curve["equity_total_pnl"],
        label="Baseline Equity (Total PnL incl fees)",
        linestyle="--",
        alpha=0.5,
        color="tab:orange",
    )
    ax_eq.plot(
        curve["entry_time"],
        curve["equity_scaled_pnl"],
        label="Scaled Equity (PnL)",
        linewidth=2,
        color="tab:blue",
    )
    ax_eq.plot(
        curve["entry_time"],
        curve["equity_scaled_total_pnl"],
        label="Scaled Equity (Total PnL incl fees)",
        linewidth=2,
        color="tab:orange",
    )
    ax_eq.axhline(initial_capital, color="gray", linestyle=":", alpha=0.7)
    ax_eq.set_title("Low-Risk Scaling Equity Curve")
    ax_eq.set_ylabel("Portfolio Value")
    ax_eq.legend()
    ax_eq.grid(True, alpha=0.3)

    ax_dd.plot(
        curve["entry_time"],
        curve["drawdown_pnl"],
        label="Baseline Drawdown (PnL)",
        linestyle="--",
        alpha=0.5,
        color="tab:blue",
    )
    ax_dd.plot(
        curve["entry_time"],
        curve["drawdown_total_pnl"],
        label="Baseline Drawdown (Total PnL incl fees)",
        linestyle="--",
        alpha=0.5,
        color="tab:orange",
    )
    ax_dd.plot(
        curve["entry_time"],
        curve["drawdown_scaled_pnl"],
        label="Scaled Drawdown (PnL)",
        linewidth=2,
        color="tab:blue",
    )
    ax_dd.plot(
        curve["entry_time"],
        curve["drawdown_scaled_total_pnl"],
        label="Scaled Drawdown (Total PnL incl fees)",
        linewidth=2,
        color="tab:orange",
    )
    ax_dd.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax_dd.set_title("Low-Risk Scaling Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Entry Time")
    ax_dd.legend()
    ax_dd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return curve


"""
Notebook usage example
----------------------

# `trade_features` should already contain:
# - entry_time
# - regime_prob
# - pnl
# - total_pnl

from src.trade_lab.scripts.regime_low_risk_scaling_example import (
    apply_low_risk_scaling,
    summarize_low_risk_scaling,
    plot_low_risk_scaling,
)

scaled_trades = apply_low_risk_scaling(
    trade_features,
    low_risk_thresh=0.10,
    low_risk_multiplier=2.0,
)

summarize_low_risk_scaling(scaled_trades)
curve = plot_low_risk_scaling(scaled_trades)

scaled_trades.loc[
    scaled_trades["is_low_risk_day"],
    ["date", "entry_time", "regime_prob", "position_multiplier", "pnl", "scaled_pnl", "total_pnl", "scaled_total_pnl"],
].head(20)
"""
