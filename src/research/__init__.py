"""Backtest analysis utilities for QuantConnect order data."""

# Data loading and transformation
from .backtest_loader import (
    build_trade_totals,
    load_orders,
)

# Exit type analysis
from .exit_analysis import (
    classify_exit_reasons,
    compute_exit_type_stats,
    get_exit_hour_distribution,
)

# Market context merging
from .market_context import (
    compare_win_loss_market_context,
    compute_range_bucketed_stats,
    merge_market_data,
)

# Performance metrics
from .performance_metrics import (
    compute_basic_stats,
    compute_distribution_stats,
    compute_expectancy,
    compute_win_loss_stats,
    get_monthly_stats,
)

# Tail risk analysis
from .tail_analysis import (
    compute_drawdown_stats,
    compute_tail_contribution,
    compute_tail_contribution_pct,
    get_best_trades,
    get_worst_trades,
    remove_worst_trades,
)

__all__ = [
    # Data loading
    "load_orders",
    "build_trade_totals",
    # Performance metrics
    "compute_basic_stats",
    "compute_win_loss_stats",
    "compute_expectancy",
    "compute_distribution_stats",
    "get_monthly_stats",
    # Tail analysis
    "compute_tail_contribution",
    "compute_tail_contribution_pct",
    "compute_drawdown_stats",
    "get_worst_trades",
    "get_best_trades",
    "remove_worst_trades",
    # Exit analysis
    "classify_exit_reasons",
    "compute_exit_type_stats",
    "get_exit_hour_distribution",
    # Market context
    "merge_market_data",
    "compute_range_bucketed_stats",
    "compare_win_loss_market_context",
]
