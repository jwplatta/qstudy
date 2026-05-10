"""Weighting schemes for Study pipeline position scaling.

All functions share the signature:
    (positions: pd.DataFrame, **cache) -> pd.DataFrame

They are designed to be used with functools.partial so they remain picklable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_equal(positions: pd.DataFrame, **cache) -> pd.DataFrame:
    """Passthrough — positions are already normalized by the position builder."""
    return positions


def apply_equal_vol(positions: pd.DataFrame, vol_window: int = 60, **cache) -> pd.DataFrame:
    """Scale positions inversely proportional to realized volatility, then renormalize.

    Assets with higher realized vol receive smaller weights.

    Args:
        positions:  Positions DataFrame (dates x tickers), +/- or long-only.
        vol_window: Lookback for realized vol calculation.
        **cache:    Study cache — must contain "returns".

    Returns:
        Rescaled positions, same shape.
    """
    returns: pd.DataFrame = cache["returns"]
    realized_vol = returns.rolling(vol_window).std().reindex(columns=positions.columns)
    inv_vol = (1.0 / realized_vol.clip(lower=1e-8)).reindex(positions.index)
    scaled = positions * inv_vol
    abs_sum = scaled.abs().sum(axis=1).replace(0, float("nan"))
    return scaled.div(abs_sum, axis=0).fillna(0.0)


def apply_equal_sharpe(positions: pd.DataFrame, window: int = 126, **cache) -> pd.DataFrame:
    """Scale positions by rolling absolute Sharpe ratio, then renormalize.

    Assets with higher recent Sharpe get larger weights. Sign is preserved from positions.

    Args:
        positions: Positions DataFrame (dates x tickers).
        window:    Lookback for rolling Sharpe calculation.
        **cache:   Study cache — must contain "returns".

    Returns:
        Rescaled positions, same shape.
    """
    returns: pd.DataFrame = cache["returns"]
    r = returns.reindex(columns=positions.columns)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std().clip(lower=1e-8)
    roll_sharpe = (roll_mean / roll_std * np.sqrt(252)).abs().reindex(positions.index)
    scaled = positions * roll_sharpe
    abs_sum = scaled.abs().sum(axis=1).replace(0, float("nan"))
    return scaled.div(abs_sum, axis=0).fillna(0.0)


def apply_optimal(
    positions: pd.DataFrame,
    window: int = 126,
    gamma: float = 1.0,
    **cache,
) -> pd.DataFrame:
    """Rolling mean-variance optimal weights (ridge-regularized, closed-form).

    For each date, solves w = (Sigma + ridge*I)^{-1} * mu for the active positions,
    then renormalizes. Falls back to equal weights when the matrix is singular or
    the lookback window has insufficient data.

    Args:
        positions: Positions DataFrame (dates x tickers), used to determine active assets.
        window:    Rolling lookback in trading days.
        gamma:     Ridge regularization multiplier on average diagonal variance.
        **cache:   Study cache — must contain "returns".

    Returns:
        Rescaled positions, same shape.
    """
    returns: pd.DataFrame = cache["returns"]
    result = positions.copy() * 0.0

    for i, date in enumerate(positions.index):
        active = positions.loc[date]
        active_tickers = active[active != 0].index.tolist()
        if not active_tickers:
            continue
        if i < window:
            result.loc[date] = positions.loc[date]
            continue

        hist = returns.iloc[max(0, i - window) : i][active_tickers].dropna(axis=1)
        if hist.shape[0] < 10 or hist.shape[1] == 0:
            result.loc[date] = positions.loc[date]
            continue

        available = hist.columns.tolist()
        mu = hist.mean().values
        sigma = hist.cov().values
        try:
            ridge = gamma * np.diag(sigma).mean() * np.eye(len(sigma))
            w = np.linalg.solve(sigma + ridge, mu)
            # Preserve sign from positions
            signs = np.sign(positions.loc[date, available].values)
            w = w * signs
            abs_total = np.abs(w).sum()
            if abs_total > 0:
                w = w / abs_total
            for j, t in enumerate(available):
                result.loc[date, t] = w[j]
        except np.linalg.LinAlgError:
            result.loc[date, available] = positions.loc[date, available]

    return result
