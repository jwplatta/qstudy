from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.qc_utils.backtest_loader import load_trades


INITIAL_CAPITAL = 50_000
REGIME_THRESHOLD = 51.698
SKIP_PROB_THRESHOLD = 0.50
LOW_RISK_THRESH = 0.10
LOW_RISK_CONTRACT_MULTIPLIER = 2.0

# Fixed logistic model from `qc/spxw_1dte_regime_forecast/main.py`
LOGIT_INTERCEPT = -3.9225339071705148
LOGIT_WEIGHTS = {
    "prior_slope": 0.15572728210666714,
    "5d_avg_range": 0.06168979627510342,
    "prior_abs_ret": 36.499736265948194,
    "gap_mag": 40.380107779126185,
}


def normalize_trade_dates(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    if "date" not in trades.columns:
        trades["date"] = trades["entry_time"].dt.date
    else:
        trades["date"] = pd.to_datetime(trades["date"]).dt.date

    return trades.sort_values("entry_time").reset_index(drop=True)


def prep_daily_ohlc(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df[datetime_col]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def filter_market_data_to_trade_window(
    trades: pd.DataFrame,
    spx_day: pd.DataFrame,
    vix_day: pd.DataFrame,
    vix9d_day: pd.DataFrame,
    lookback_calendar_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_date = trades["date"].min() - timedelta(days=lookback_calendar_days)
    end_date = trades["date"].max()

    spx_f = spx_day[(spx_day["date"] >= start_date) & (spx_day["date"] <= end_date)].copy()
    vix_f = vix_day[(vix_day["date"] >= start_date) & (vix_day["date"] <= end_date)].copy()
    vix9d_f = vix9d_day[(vix9d_day["date"] >= start_date) & (vix9d_day["date"] <= end_date)].copy()

    return spx_f, vix_f, vix9d_f


def compute_spx_features(spx_day: pd.DataFrame) -> pd.DataFrame:
    spx = spx_day.copy()

    spx["range"] = spx["high"] - spx["low"]
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["5d_avg_range"] = spx["range"].shift(1).rolling(5).mean()
    spx["prior_abs_ret"] = spx["ret"].shift(1).abs()
    spx["gap_mag"] = ((spx["open"] - spx["close"].shift(1)) / spx["close"].shift(1)).abs()

    return spx[["date", "5d_avg_range", "prior_abs_ret", "gap_mag"]]


def compute_vix_features(vix_day: pd.DataFrame, vix9d_day: pd.DataFrame) -> pd.DataFrame:
    vix = vix_day.copy()
    vix9d = vix9d_day.copy()

    vix["prior_vix_close"] = vix["close"].shift(1)
    vix9d["prior_vix9d_close"] = vix9d["close"].shift(1)

    out = vix[["date", "prior_vix_close"]].merge(
        vix9d[["date", "prior_vix9d_close"]],
        on="date",
        how="inner",
    )
    out["prior_slope"] = out["prior_vix9d_close"] - out["prior_vix_close"]

    return out[["date", "prior_slope"]]


def build_trade_feature_table(
    trades: pd.DataFrame,
    spx_day: pd.DataFrame,
    vix_day: pd.DataFrame,
    vix9d_day: pd.DataFrame,
) -> pd.DataFrame:
    spx_features = compute_spx_features(spx_day)
    vix_features = compute_vix_features(vix_day, vix9d_day)
    features = spx_features.merge(vix_features, on="date", how="inner")

    return trades.merge(features, on="date", how="left")


def compute_regime_probability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    logit = (
        LOGIT_INTERCEPT
        + LOGIT_WEIGHTS["prior_slope"] * df["prior_slope"]
        + LOGIT_WEIGHTS["5d_avg_range"] * df["5d_avg_range"]
        + LOGIT_WEIGHTS["prior_abs_ret"] * df["prior_abs_ret"]
        + LOGIT_WEIGHTS["gap_mag"] * df["gap_mag"]
    )

    df["regime_prob"] = 1.0 / (1.0 + np.exp(-logit))
    df["skip_trade"] = df["regime_prob"] >= SKIP_PROB_THRESHOLD
    df["keep_trade"] = ~df["skip_trade"]

    return df


def filter_trades(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "prior_slope",
        "5d_avg_range",
        "prior_abs_ret",
        "gap_mag",
        "regime_prob",
    ]
    return (
        df.dropna(subset=required)
        .loc[lambda x: x["keep_trade"]]
        .sort_values("entry_time")
        .reset_index(drop=True)
    )


def apply_contract_multiplier(
    df: pd.DataFrame,
    low_risk_thresh: float = LOW_RISK_THRESH,
    contract_multiplier: float = LOW_RISK_CONTRACT_MULTIPLIER,
) -> pd.DataFrame:
    out = df.copy().sort_values("entry_time").reset_index(drop=True)
    out["base_pnl"] = out["pnl"]
    out["base_total_pnl"] = out["total_pnl"]
    out["contract_multiplier"] = 1.0
    out["is_low_risk_day"] = out["regime_prob"] <= low_risk_thresh
    out.loc[out["is_low_risk_day"], "contract_multiplier"] = contract_multiplier
    out["pnl"] = out["base_pnl"] * out["contract_multiplier"]
    out["total_pnl"] = out["base_total_pnl"] * out["contract_multiplier"]
    return out


def add_equity_and_drawdown(
    df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL
) -> pd.DataFrame:
    df = df.copy().sort_values("entry_time").reset_index(drop=True)

    df["equity_pnl"] = initial_capital + df["pnl"].cumsum()
    df["equity_total_pnl"] = initial_capital + df["total_pnl"].cumsum()

    df["peak_pnl"] = df["equity_pnl"].cummax()
    df["peak_total_pnl"] = df["equity_total_pnl"].cummax()

    df["drawdown_pnl"] = df["equity_pnl"] / df["peak_pnl"] - 1.0
    df["drawdown_total_pnl"] = df["equity_total_pnl"] / df["peak_total_pnl"] - 1.0

    return df


def summarize_filtering(all_trades: pd.DataFrame, filtered_trades: pd.DataFrame) -> None:
    print("All trades:", len(all_trades))
    print("Kept trades:", len(filtered_trades))
    print("Skipped trades:", len(all_trades) - len(filtered_trades))
    if "is_low_risk_day" in filtered_trades.columns:
        print("Low-risk scaled trades:", int(filtered_trades["is_low_risk_day"].sum()))
    print()

    print("Baseline final equity (PnL):", INITIAL_CAPITAL + all_trades["pnl"].sum())
    print("Baseline final equity (Total PnL):", INITIAL_CAPITAL + all_trades["total_pnl"].sum())
    print("Filtered final equity (PnL):", INITIAL_CAPITAL + filtered_trades["pnl"].sum())
    print(
        "Filtered final equity (Total PnL):", INITIAL_CAPITAL + filtered_trades["total_pnl"].sum()
    )


def plot_probability_distribution(trade_features: pd.DataFrame) -> None:
    plot_df = trade_features.dropna(subset=["regime_prob"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(plot_df["regime_prob"], bins=30, alpha=0.8, color="tab:blue")
    axes[0].axvline(SKIP_PROB_THRESHOLD, color="red", linestyle="--", label="Skip threshold")
    axes[0].set_title("Regime Probability Distribution")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Trade Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    kept = plot_df.loc[~plot_df["skip_trade"]]
    skipped = plot_df.loc[plot_df["skip_trade"]]

    axes[1].scatter(
        kept["entry_time"],
        kept["regime_prob"],
        s=20,
        alpha=0.7,
        label="Kept",
        color="tab:blue",
    )
    axes[1].scatter(
        skipped["entry_time"],
        skipped["regime_prob"],
        s=28,
        alpha=0.85,
        label="Skipped",
        color="tab:red",
    )
    axes[1].axhline(SKIP_PROB_THRESHOLD, color="red", linestyle="--")
    axes[1].set_title("Predicted Probability by Trade Date")
    axes[1].set_xlabel("Entry Time")
    axes[1].set_ylabel("Predicted Probability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_equity_and_drawdown(
    filtered_trades: pd.DataFrame,
    all_trades: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    filtered_curve = add_equity_and_drawdown(filtered_trades, initial_capital)
    baseline_curve = (
        None if all_trades is None else add_equity_and_drawdown(all_trades, initial_capital)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_eq, ax_dd = axes

    if baseline_curve is not None:
        ax_eq.plot(
            baseline_curve["entry_time"],
            baseline_curve["equity_pnl"],
            label="Baseline Equity (PnL)",
            alpha=0.35,
            linestyle="--",
            color="tab:blue",
        )
        ax_eq.plot(
            baseline_curve["entry_time"],
            baseline_curve["equity_total_pnl"],
            label="Baseline Equity (Total PnL incl fees)",
            alpha=0.35,
            linestyle="--",
            color="tab:orange",
        )

    ax_eq.plot(
        filtered_curve["entry_time"],
        filtered_curve["equity_pnl"],
        label="Filtered Equity (PnL)",
        linewidth=2,
        color="tab:blue",
    )
    ax_eq.plot(
        filtered_curve["entry_time"],
        filtered_curve["equity_total_pnl"],
        label="Filtered Equity (Total PnL incl fees)",
        linewidth=2,
        color="tab:orange",
    )
    ax_eq.axhline(initial_capital, color="gray", linestyle=":", alpha=0.7)
    ax_eq.set_title("Equity Curve")
    ax_eq.set_ylabel("Portfolio Value")
    ax_eq.legend()
    ax_eq.grid(True, alpha=0.3)

    if baseline_curve is not None:
        ax_dd.plot(
            baseline_curve["entry_time"],
            baseline_curve["drawdown_pnl"],
            alpha=0.35,
            linestyle="--",
            color="tab:blue",
        )
        ax_dd.plot(
            baseline_curve["entry_time"],
            baseline_curve["drawdown_total_pnl"],
            alpha=0.35,
            linestyle="--",
            color="tab:orange",
        )

    ax_dd.plot(
        filtered_curve["entry_time"],
        filtered_curve["drawdown_pnl"],
        label="Filtered Drawdown (PnL)",
        linewidth=2,
        color="tab:blue",
    )
    ax_dd.plot(
        filtered_curve["entry_time"],
        filtered_curve["drawdown_total_pnl"],
        label="Filtered Drawdown (Total PnL incl fees)",
        linewidth=2,
        color="tab:orange",
    )
    ax_dd.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Entry Time")
    ax_dd.legend()
    ax_dd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return filtered_curve


def run_regime_filter_backtest(
    trades: pd.DataFrame,
    spx_day: pd.DataFrame,
    vix_day: pd.DataFrame,
    vix9d_day: pd.DataFrame,
    low_risk_thresh: float | None = None,
    contract_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = normalize_trade_dates(trades)
    spx_day = prep_daily_ohlc(spx_day)
    vix_day = prep_daily_ohlc(vix_day)
    vix9d_day = prep_daily_ohlc(vix9d_day)

    spx_day, vix_day, vix9d_day = filter_market_data_to_trade_window(
        trades,
        spx_day,
        vix_day,
        vix9d_day,
        lookback_calendar_days=14,
    )

    trade_features = build_trade_feature_table(trades, spx_day, vix_day, vix9d_day)
    trade_features = compute_regime_probability(trade_features)

    valid_trades = trade_features.dropna(subset=["regime_prob"]).copy()
    filtered_trades = filter_trades(trade_features)

    if low_risk_thresh is not None and contract_multiplier != 1.0:
        filtered_trades = apply_contract_multiplier(
            filtered_trades,
            low_risk_thresh=low_risk_thresh,
            contract_multiplier=contract_multiplier,
        )

    summarize_filtering(valid_trades, filtered_trades)
    plot_probability_distribution(valid_trades)
    plot_equity_and_drawdown(
        filtered_trades, all_trades=valid_trades, initial_capital=INITIAL_CAPITAL
    )

    return trade_features, filtered_trades


if __name__ == "__main__":
    trades = load_trades("research/data/baseline_v1_trades_02_13_2026.csv")
    spx_day = pd.read_csv("research/data/SPX_day_1980-01-01_2026-03-24.csv", header=0)
    vix_day = pd.read_csv("research/data/VIX_day_1980-01-01_2026-03-24.csv", header=0)
    vix9d_day = pd.read_csv("research/data/VIX9D_day_2000-01-01_2026-03-24.csv", header=0)

    trade_features, filtered_trades = run_regime_filter_backtest(
        trades=trades,
        spx_day=spx_day,
        vix_day=vix_day,
        vix9d_day=vix9d_day,
        low_risk_thresh=LOW_RISK_THRESH,
        contract_multiplier=LOW_RISK_CONTRACT_MULTIPLIER,
    )
