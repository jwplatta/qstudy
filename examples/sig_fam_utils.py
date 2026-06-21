from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

SignalFn = Callable[..., pd.DataFrame]
FilterFn = Callable[..., pd.DataFrame]
ScalerFn = Callable[..., pd.DataFrame]
PartnersProvider = Callable[[], dict[int, dict[str, list[str]]]]
SectorEtfMapProvider = Callable[[], dict[str, str]]


@dataclass(frozen=True)
class SleeveSpec:
    name: str
    signal_fn: SignalFn
    conditioning_filter: FilterFn | None
    rebalance_every: int
    risk_scalers: list[ScalerFn]
    use_factor_model: bool
    use_etf_resid: bool
    needs_resid_cache: bool


SIGNAL_POOL_SLEEVE_NAMES = [
    # --- Mean Reversion (5) ---
    "cumret_spread_20_252__r5__vol_20_60__cond__none",
    "cumret_spread_20_252__r10__vol_20_60__cond__none",
    "cumret_spread_20_252__r5__none__cond__none",
    "cumret_spread_20_252__r5__trend_20_100_h__cond__none",
    "zscore_rev_20_252__r5__vol_20_60__cond__none",
    # --- Residual MR (5) ---
    "etf_factor_resid_mr_5d__r10__trend_50_200__cond__none",
    "factor_model_resid_mr_10d__r10__trend_20_100__cond__none",
    "factor_model_resid_mr_10d__r5__trend_20_100__cond__none",
    "resid_zscore_w15_w10__r10__trend_20_100__cond__none",
    "resid_zscore_w15_w10__r10__trend_20_100_h__cond__none",
    # --- Distance Pairs (4) ---
    "dist_mr_k1_z60__r21__none__cond__none",
    "dist_mr_k1_z20__r21__none__cond__none",
    "dist_mr_k1_z60__r10__none__cond__none",
    "dist_mr_k3_z60__r21__none__cond__none",
    # --- Event / Gap Accumulation (4) ---
    "gap_accum_3d__r21__trend_20_100__cond__none",
    "gap_accum_3d__r21__trend_20_100_h__cond__none",
    "gap_accum_2d__r10__breadth_20_q30__cond__none",
    "resid_gap_accum_5d__r10__disp_60_q75__cond__none",
    # --- Momentum (5) ---
    "sharpe_mom_252d__r21__none__cond__none",
    "ma_dist_50_200__r21__none__cond__none",
    "ma_dist_20_200__r21__none__cond__none",
    "mom_252d__r21__vol_20_60__cond__none",
    "mom_252d__r21__breadth_50__cond__none",
    # --- Monotonicity (4) ---
    "monoton_skip_252d__r21__vol_10_60__cond__none",
    "monoton_skip_252d__r21__trend_50_200__cond__none",
    "monoton_skip_252d__r21__trend_20_100__cond__none",
    "monoton_w252d__r21__breadth_50__cond__none",
    # --- Residual Momentum (3) ---
    "resid_mom_252d__r21__vol_20_60__cond__none",
    "resid_mom_252d__r21__vol_10_60__cond__none",
    "resid_mom_252d__r21__trend_10_60__cond__none",
    # --- Sharpe-Residual Momentum (2) ---
    "sharpe_resid_mom_252d_skip5__r21__vol_20_60__cond__none",
    "sharpe_resid_mom_252d_skip5__r21__vol_20_100__cond__none",
    # --- Sector-Relative (3) ---
    "sector_rel_mr_20d__r10__vol_20_100__cond__none",
    "sector_rel_sharpe_252d_skip5__r21__breadth_50__cond__none",
    "sector_rel_mom_120d__r21__trend_20_100__cond__none",
    # --- Vol-Trend (5) ---
    "vol_regime_ret_10_90_r5__r5__none__cond__none",
    "vol_regime_ret_10_90_r5__r10__none__cond__none",
    "ivol_explosion_5_60_p90__r10__none__cond__none",
    "ivol_accel_20_90__r21__none__cond__none",
    "ivol_accel_20_120__r21__none__cond__none",
    # --- Beta Momentum ---
    "beta_momentum_60_252d__r21__breadth_40__cond__none",
    "beta_momentum_60_252d__r21__none__cond__none",
    "beta_momentum_60_252d__r21__vol_10_60__cond__none",
    "beta_momentum_60_252d__r21__trend_50_200__cond__none",
    # --- Bear Narrow-Breadth ---
    # vol_accel_20_120d: best vol compression signal, consistent across years, positive 2022
    "vol_accel_20_120d__r10__none__cond__breadth_lt40",
    "vol_accel_20_120d__r10__trend_50_200_mr__cond__breadth_lt40",
    "vol_accel_20_120d__r10__trend_20_100_mr__cond__breadth_lt40",
    # vol_accel_5_60d: faster variant, also positive 2022, consistent
    "vol_accel_5_60d__r21__vol_20_60_up__cond__breadth_lt40",
    "vol_accel_5_60d__r21__trend_50_200_mr__cond__breadth_lt40",
    "vol_accel_5_60d__r21__trend_20_100_mr__cond__breadth_lt40",
    # bear_reversal: short-term MR gated on bear+narrow. Zero negative years, mean Sharpe >1.
    "bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40",
    "bear_reversal_20d__r21__none__cond__bear_narrow_lt40",
    # --- Narrow-Breadth ---
    # Active only when pct stocks above 200d MA < 50% (covers 2022 bear + 2023+ narrow bull)
    "beta_momentum_60_252d__r21__none__cond__breadth_lt50",
    "beta_momentum_60_252d__r21__trend_50_200_mom__cond__breadth_lt50",
    "sharpe_mom_120d__r5__trend_50_200_mom__cond__breadth_lt50",
    "etf_mom_120d__r5__trend_50_200_mr__cond__disp_60_q60",
    "etf_mom_120d__r5__trend_50_200_mom__cond__disp_60_q60",
    "etf_mom_120d__r5__none__cond__disp_60_q60",
    "etf_mom_120d__r21__trend_50_200_mr__cond__breadth_lt50",
    "etf_sharpe_mom_120d__r21__trend_50_200_mr__cond__breadth_lt50",
    "etf_mom_120d__r21__none__cond__breadth_lt50",
    "beta_momentum_60_252d__r10__none__cond__breadth_lt50",
    "sector_rel_mom_active_60d__r21__trend_50_200_mom__cond__breadth_lt50",
    # --- Bull Narrow-Breadth ---
    # low_vol_mom_120d: best from bull narrow sweep. mean +0.95, only 1 negative year (-0.42 in 2022)
    # consistent across 2017(+2.46), 2019(+1.38), 2020(+1.52), 2023(+1.91)
    "low_vol_mom_120d__r21__vol_20_60__cond__breadth_lt50",
    "low_vol_mom_120d__r21__trend_20_100_mom__cond__breadth_lt50",
    "low_vol_mom_120d__r21__none__cond__breadth_lt50",
    "low_vol_mom_120d__r5__trend_50_200_mom__cond__breadth_lt50",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_from_first_valid(frame: pd.DataFrame) -> pd.DataFrame:
    first_valid = frame.bfill().iloc[0].replace(0.0, np.nan)
    return frame.div(first_valid, axis=1)


def _sector_factor_frame(
    active_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    get_sector_etf_map: SectorEtfMapProvider,
) -> pd.DataFrame:
    etf_map = get_sector_etf_map()
    sector_df = pd.DataFrame(
        index=active_returns.index,
        columns=active_returns.columns,
        dtype=float,
    )
    for ticker in active_returns.columns:
        etf = etf_map.get(ticker, "SPY")
        sector_df[ticker] = (
            factor_returns[etf].reindex(active_returns.index).fillna(0.0)
            if etf in factor_returns.columns
            else 0.0
        )
    return sector_df


# ---------------------------------------------------------------------------
# Signal factories — Mean Reversion
# ---------------------------------------------------------------------------


def make_mr(window: int) -> SignalFn:
    def mr(**cache):
        return -cache["_active_returns"].rolling(window).mean()

    mr.__name__ = f"mr_{window}d"
    return mr


def make_zscore_rev(short_window: int, long_window: int) -> SignalFn:
    def zscore_rev(**cache):
        returns = cache["_active_returns"]
        mean_long = returns.rolling(long_window).mean()
        std_long = returns.rolling(long_window).std().clip(lower=1e-8)
        return -(returns.rolling(short_window).mean() - mean_long) / std_long

    zscore_rev.__name__ = f"zscore_rev_{short_window}_{long_window}"
    return zscore_rev


def make_cumret_spread(short_window: int, long_window: int) -> SignalFn:
    """Cumulative-return spread: -(fast_mean - slow_mean)."""

    def cumret_spread(**cache):
        r = cache["_active_returns"]
        return -(r.rolling(short_window).mean() - r.rolling(long_window).mean())

    cumret_spread.__name__ = f"cumret_spread_{short_window}_{long_window}"
    return cumret_spread


# ---------------------------------------------------------------------------
# Signal factories — Residual Mean Reversion
# ---------------------------------------------------------------------------


def make_residual_mr(window: int, name: str) -> SignalFn:
    def residual_mr(**cache):
        return -cache["residual_returns"].rolling(window).mean()

    residual_mr.__name__ = name
    return residual_mr


def make_resid_zscore_w15(window: int) -> SignalFn:
    """Tight-winsorized (±1.5) z-score on residual returns."""

    def resid_zscore_w15(**cache):
        r = cache["residual_returns"]
        mu = r.rolling(window).mean()
        sigma = r.rolling(window).std().clip(lower=1e-8)
        return -((r - mu) / sigma).clip(-1.5, 1.5)

    resid_zscore_w15.__name__ = f"resid_zscore_w15_w{window}"
    return resid_zscore_w15


# ---------------------------------------------------------------------------
# Signal factories — Distance-Pairs Mean Reversion
# ---------------------------------------------------------------------------


def make_dist_mr(zw: int, get_distance_partners: PartnersProvider) -> SignalFn:
    """k=3 nearest partners distance MR."""

    def dist_mr(**cache):
        returns = cache["_active_returns"]
        partners = get_distance_partners()[zw]
        price = (1 + returns).cumprod()
        norm = normalize_from_first_valid(price)
        spread = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for ticker in returns.columns:
            peers = [peer for peer in partners.get(ticker, []) if peer in returns.columns]
            if peers:
                spread[ticker] = norm[ticker] - norm[peers].mean(axis=1)
        mean_spread = spread.rolling(zw).mean()
        std_spread = spread.rolling(zw).std().clip(lower=1e-8)
        return -((spread - mean_spread) / std_spread).clip(-2, 2)

    dist_mr.__name__ = f"dist_mr_k3_z{zw}"
    return dist_mr


def make_dist_mr_k1(zw: int, get_distance_partners: PartnersProvider) -> SignalFn:
    """k=1 nearest partner distance MR."""

    def dist_mr_k1(**cache):
        returns = cache["_active_returns"]
        partners = get_distance_partners()[zw]
        price = (1 + returns).cumprod()
        norm = normalize_from_first_valid(price)
        spread = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for ticker in returns.columns:
            peers = [peer for peer in partners.get(ticker, []) if peer in returns.columns]
            if peers:
                # k=1: take only the first (closest) partner
                spread[ticker] = norm[ticker] - norm[[peers[0]]].mean(axis=1)
            else:
                spread[ticker] = np.nan
        mu = spread.rolling(zw).mean()
        sigma = spread.rolling(zw).std().clip(lower=1e-8)
        return -((spread - mu) / sigma).clip(-2, 2)

    dist_mr_k1.__name__ = f"dist_mr_k1_z{zw}"
    return dist_mr_k1


# ---------------------------------------------------------------------------
# Signal factories — Event / Gap Accumulation
# ---------------------------------------------------------------------------


def make_gap_accum(window: int) -> SignalFn:
    """Gap accumulation: -(rolling max return over window)."""

    def gap_accum(**cache):
        r = cache["_active_returns"]
        return -r.rolling(window).max()

    gap_accum.__name__ = f"gap_accum_{window}d"
    return gap_accum


def make_resid_gap_accum(window: int) -> SignalFn:
    """Residualized gap accumulation: -(rolling max residual return over window)."""

    def resid_gap_accum(**cache):
        r = cache["residual_returns"]
        return -r.rolling(window).max()

    resid_gap_accum.__name__ = f"resid_gap_accum_{window}d"
    return resid_gap_accum


def make_resid_gap_reversion() -> SignalFn:
    def resid_gap_reversion(**cache):
        return -cache["residual_returns"].shift(1)

    resid_gap_reversion.__name__ = "resid_gap_reversion"
    return resid_gap_reversion


# ---------------------------------------------------------------------------
# Signal factories — Momentum
# ---------------------------------------------------------------------------


def make_sharpe_mom(window: int) -> SignalFn:
    """Rolling Sharpe-ratio momentum on active returns."""

    def sharpe_mom(**cache):
        r = cache["_active_returns"]
        mu = r.rolling(window).mean()
        sigma = r.rolling(window).std().clip(lower=1e-8)
        return mu / sigma

    sharpe_mom.__name__ = f"sharpe_mom_{window}d"
    return sharpe_mom


def make_low_vol_mom(window: int) -> SignalFn:
    """Low-volatility momentum: (mu/sigma) / sigma = mu / sigma^2.
    Rewards consistent low-vol winners; penalizes high-vol even if strong momentum."""

    def low_vol_mom(**cache):
        r = cache["_active_returns"]
        mu = r.rolling(window).mean()
        sigma = r.rolling(window).std().clip(lower=1e-8)
        return mu / (sigma * sigma)

    low_vol_mom.__name__ = f"low_vol_mom_{window}d"
    return low_vol_mom


def make_ma_dist(fast: int, slow: int) -> SignalFn:
    """Moving-average distance: price.rolling(fast).mean() / price.rolling(slow).mean() - 1."""

    def ma_dist(**cache):
        r = cache["_active_returns"]
        price = (1 + r).cumprod()
        return price.rolling(fast).mean() / price.rolling(slow).mean() - 1

    ma_dist.__name__ = f"ma_dist_{fast}_{slow}"
    return ma_dist


def make_mom(window: int) -> SignalFn:
    """Raw cumulative return momentum."""

    def mom(**cache):
        return cache["_active_returns"].rolling(window).sum()

    mom.__name__ = f"mom_{window}d"
    return mom


# ---------------------------------------------------------------------------
# Signal factories — Monotonicity
# ---------------------------------------------------------------------------


def make_monoton_120d() -> SignalFn:
    """Legacy 120d monotonicity (kept for backward compat)."""

    def monoton_120d(**cache):
        returns = cache["_active_returns"]
        mean_120 = returns.rolling(120).mean()
        same_sign = (returns.gt(0) == mean_120.gt(0)).astype(float)
        return same_sign.rolling(120).mean() * mean_120.abs()

    monoton_120d.__name__ = "monoton_120d"
    return monoton_120d


def make_monoton_skip_252d() -> SignalFn:
    """Skip-1-week weighted consistency over 252d: shifts returns by 5 days first."""

    def monoton_skip_252d(**cache):
        r = cache["_active_returns"].shift(5)
        mu = r.rolling(252).mean()
        return (r.gt(0) == mu.gt(0)).rolling(252).mean() * mu.abs()

    monoton_skip_252d.__name__ = "monoton_skip_252d"
    return monoton_skip_252d


def make_monoton_w252d() -> SignalFn:
    """Weighted consistency over 252d (unshifted)."""

    def monoton_w252d(**cache):
        r = cache["_active_returns"]
        mu = r.rolling(252).mean()
        return (r.gt(0) == mu.gt(0)).rolling(252).mean() * mu.abs()

    monoton_w252d.__name__ = "monoton_w252d"
    return monoton_w252d


# ---------------------------------------------------------------------------
# Signal factories — Residual Momentum
# ---------------------------------------------------------------------------


def make_resid_mom(window: int) -> SignalFn:
    """Residual momentum: rolling mean of factor-model-residualized returns."""

    def resid_mom(**cache):
        return cache["residual_returns"].rolling(window).mean()

    resid_mom.__name__ = f"resid_mom_{window}d"
    return resid_mom


def make_sharpe_resid_mom(window: int, skip_days: int) -> SignalFn:
    """Sharpe-scaled residual momentum with optional skip."""

    def sharpe_resid_mom(**cache):
        r = cache["residual_returns"]
        mu = r.shift(skip_days).rolling(window).mean()
        sigma = r.shift(skip_days).rolling(window).std().clip(lower=1e-8)
        return mu / sigma

    suffix = f"_skip{skip_days}" if skip_days else ""
    sharpe_resid_mom.__name__ = f"sharpe_resid_mom_{window}d{suffix}"
    return sharpe_resid_mom


# ---------------------------------------------------------------------------
# Signal factories — Sector-Relative
# ---------------------------------------------------------------------------


def make_sector_rel_mr(window: int, get_sector_etf_map: SectorEtfMapProvider) -> SignalFn:
    def sector_rel_mr(**cache):
        returns = cache["_active_returns"]
        factor_returns = cache["factor_returns"]
        sector_returns = _sector_factor_frame(returns, factor_returns, get_sector_etf_map)
        return -(returns - sector_returns).rolling(window).mean()

    sector_rel_mr.__name__ = f"sector_rel_mr_{window}d"
    return sector_rel_mr


def make_sector_rel_zscore_rev(
    short_window: int,
    long_window: int,
    get_sector_etf_map: SectorEtfMapProvider,
) -> SignalFn:
    def sector_rel_zscore(**cache):
        returns = cache["_active_returns"]
        factor_returns = cache["factor_returns"]
        sector_returns = _sector_factor_frame(returns, factor_returns, get_sector_etf_map)
        rel = returns - sector_returns
        mean_long = rel.rolling(long_window).mean()
        std_long = rel.rolling(long_window).std().clip(lower=1e-8)
        return -(rel.rolling(short_window).mean() - mean_long) / std_long

    sector_rel_zscore.__name__ = f"sector_rel_zscore_{short_window}_{long_window}"
    return sector_rel_zscore


def make_sector_rel_sharpe(
    window: int,
    skip_days: int,
    get_sector_etf_map: SectorEtfMapProvider,
) -> SignalFn:
    """Sharpe-normalized sector-relative momentum (stock-sector spread / spread std)."""

    def sector_rel_sharpe(**cache):
        r = cache["residual_returns"]
        factor_returns = cache["factor_returns"]
        sector_r = _sector_factor_frame(r, factor_returns, get_sector_etf_map)
        spread = r - sector_r
        mu = spread.shift(skip_days).rolling(window).mean()
        sigma = spread.shift(skip_days).rolling(window).std().clip(lower=1e-8)
        return mu / sigma

    suffix = f"_skip{skip_days}" if skip_days else ""
    sector_rel_sharpe.__name__ = f"sector_rel_sharpe_{window}d{suffix}"
    return sector_rel_sharpe


def make_sector_rel_mom(
    window: int,
    get_sector_etf_map: SectorEtfMapProvider,
) -> SignalFn:
    """Sector-relative momentum: (stock - sector_etf).rolling(window).mean()."""

    def sector_rel_mom(**cache):
        r = cache["residual_returns"]
        factor_returns = cache["factor_returns"]
        sector_r = _sector_factor_frame(r, factor_returns, get_sector_etf_map)
        return (r - sector_r).rolling(window).mean()

    sector_rel_mom.__name__ = f"sector_rel_mom_{window}d"
    return sector_rel_mom


# ---------------------------------------------------------------------------
# Signal factories — Vol-Trend
# ---------------------------------------------------------------------------


def make_ivol_accel(fast: int, slow: int) -> SignalFn:
    """IVOL acceleration: -(fast_resid_vol - slow_resid_vol). Positive = vol compressing."""

    def ivol_accel(**cache):
        r = cache.get("residual_returns", cache["_active_returns"])
        return -(r.rolling(fast).std() - r.rolling(slow).std())

    ivol_accel.__name__ = f"ivol_accel_{fast}_{slow}"
    return ivol_accel


def make_vol_accel(fast: int, slow: int) -> SignalFn:
    """Vol acceleration on active returns: -(fast_vol - slow_vol). Positive = vol compressing."""

    def vol_accel(**cache):
        r = cache["_active_returns"]
        return -(r.rolling(fast).std() - r.rolling(slow).std())

    vol_accel.__name__ = f"vol_accel_{fast}_{slow}d"
    return vol_accel


def make_bear_reversal(window: int) -> SignalFn:
    """Short-term mean reversion on active returns: -rolling_mean. Bear regime MR."""

    def bear_reversal(**cache):
        return -cache["_active_returns"].rolling(window).mean()

    bear_reversal.__name__ = f"bear_reversal_{window}d"
    return bear_reversal


def make_ivol_explosion(fast: int, slow: int, percentile: float) -> SignalFn:
    """IVOL acceleration active only in top percentile of vol-ratio cross-section."""

    def ivol_explosion(**cache):
        r = cache.get("residual_returns", cache["_active_returns"])
        fast_vol = r.rolling(fast).std()
        slow_vol = r.rolling(slow).std().clip(lower=1e-8)
        vol_ratio = fast_vol / slow_vol
        threshold = vol_ratio.quantile(percentile, axis=1)
        in_explosion = vol_ratio.ge(threshold, axis=0)
        accel = -(fast_vol - slow_vol)
        return accel.where(in_explosion)

    pct_str = str(int(percentile * 100))
    ivol_explosion.__name__ = f"ivol_explosion_{fast}_{slow}_p{pct_str}"
    return ivol_explosion


def make_vol_regime_ret(fast: int, slow: int, ret_window: int) -> SignalFn:
    """Vol-regime × residual return: sign(ivol_accel) * |r.rolling(ret_window).mean()|."""

    def vol_regime_ret(**cache):
        r = cache.get("residual_returns", cache["_active_returns"])
        accel = -(r.rolling(fast).std() - r.rolling(slow).std())
        direction = accel.apply(np.sign)
        magnitude = r.rolling(ret_window).mean().abs()
        return direction * magnitude

    vol_regime_ret.__name__ = f"vol_regime_ret_{fast}_{slow}_r{ret_window}"
    return vol_regime_ret


def make_beta_momentum(fast: int, slow: int) -> SignalFn:
    """Beta momentum: change in rolling beta (fast vs slow window).

    Stocks whose beta is falling (becoming more defensive) get a positive
    signal (long); stocks whose beta is rising (becoming more aggressive)
    get a negative signal (short). Negated so low-and-falling beta → long.
    """

    def _rolling_beta_single(s: pd.Series, mkt: pd.Series, window: int) -> pd.Series:
        mkt_var = mkt.rolling(window).var().clip(lower=1e-10)
        cov = (s * mkt).rolling(window).mean() - s.rolling(window).mean() * mkt.rolling(
            window
        ).mean()
        return cov / mkt_var

    def beta_momentum(**cache):
        r = cache["_active_returns"]
        bm = cache.get("benchmark")
        if bm is None:
            return pd.DataFrame(np.nan, index=r.index, columns=r.columns)
        mkt = bm.reindex(r.index).fillna(0.0)
        beta_fast = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        beta_slow = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        for col in r.columns:
            s = r[col].fillna(0.0)
            beta_fast[col] = _rolling_beta_single(s, mkt, fast)
            beta_slow[col] = _rolling_beta_single(s, mkt, slow)
        # Negate: falling beta (fast < slow) → positive signal → long
        return -(beta_fast - beta_slow)

    beta_momentum.__name__ = f"beta_momentum_{fast}_{slow}d"
    return beta_momentum


# ---------------------------------------------------------------------------
# Signal factories — Sector ETF Momentum (narrow-breadth)
# ---------------------------------------------------------------------------


def make_etf_mom(window: int, get_sector_etf_map: SectorEtfMapProvider) -> SignalFn:
    """Sector ETF momentum: assigns each stock its sector ETF's trailing mean return."""

    def etf_mom(**cache):
        r = cache["_active_returns"]
        factor_returns = cache["factor_returns"]
        sector_df = _sector_factor_frame(r, factor_returns, get_sector_etf_map)
        return sector_df.rolling(window).mean()

    etf_mom.__name__ = f"etf_mom_{window}d"
    return etf_mom


def make_etf_sharpe_mom(window: int, get_sector_etf_map: SectorEtfMapProvider) -> SignalFn:
    """Sharpe-ratio-weighted sector ETF momentum."""

    def etf_sharpe_mom(**cache):
        r = cache["_active_returns"]
        factor_returns = cache["factor_returns"]
        sector_df = _sector_factor_frame(r, factor_returns, get_sector_etf_map)
        mu = sector_df.rolling(window).mean()
        sigma = sector_df.rolling(window).std().clip(lower=1e-8)
        return mu / sigma

    etf_sharpe_mom.__name__ = f"etf_sharpe_mom_{window}d"
    return etf_sharpe_mom


def make_sector_rel_mom_active(window: int, get_sector_etf_map: SectorEtfMapProvider) -> SignalFn:
    """Sector-relative momentum using active (non-residualized) returns."""

    def sector_rel_mom_active(**cache):
        r = cache["_active_returns"]
        factor_returns = cache["factor_returns"]
        sector_df = _sector_factor_frame(r, factor_returns, get_sector_etf_map)
        return (r - sector_df).rolling(window).mean()

    sector_rel_mom_active.__name__ = f"sector_rel_mom_active_{window}d"
    return sector_rel_mom_active


# ---------------------------------------------------------------------------
# Filter factories
# ---------------------------------------------------------------------------


def filter_residual_dispersion_high_20_q75(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    resid = cache["residual_returns"]
    disp = resid.std(axis=1).dropna()
    disp = disp.rolling(20, min_periods=20).mean()
    thresh = disp.rolling(252, min_periods=252).quantile(0.75)
    mask = disp.gt(thresh).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_residual_dispersion_high_20_q75.__name__ = "residual_dispersion_high_20_q75"


def filter_dispersion_high_60_q75(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    disp = cache["returns"].std(axis=1).dropna()
    disp = disp.rolling(60, min_periods=60).mean()
    thresh = disp.rolling(252, min_periods=252).quantile(0.75)
    mask = disp.gt(thresh).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_dispersion_high_60_q75.__name__ = "dispersion_high_60_q75"


def filter_dispersion_high_60_q60(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    """Active when 60d smoothed cross-sectional dispersion is above its 60th percentile."""
    disp = cache["returns"].std(axis=1).dropna()
    disp = disp.rolling(60, min_periods=60).mean()
    thresh = disp.rolling(252, min_periods=126).quantile(0.60)
    mask = disp.gt(thresh).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_dispersion_high_60_q60.__name__ = "dispersion_high_60_q60"


def filter_breadth_weak_40(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    prices = (1 + cache["returns"]).cumprod()
    ma_200 = prices.rolling(200).mean()
    pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
    mask = pct_above.lt(0.40).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_breadth_weak_40.__name__ = "breadth_weak_40"


def filter_bear_narrow_lt40(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    """Active when breadth < 40% AND SPY 50d MA < 200d MA (strict bear + narrow gate)."""
    prices = (1 + cache["returns"]).cumprod()
    ma_200 = prices.rolling(200).mean()
    pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
    narrow = pct_above.lt(0.40).reindex(signal.index).fillna(False)
    bm = cache.get("benchmark")
    if bm is not None:
        spy_price = (1 + bm).cumprod()
        downtrend = spy_price.rolling(50).mean().lt(spy_price.rolling(200).mean())
        downtrend = downtrend.reindex(signal.index).fillna(False)
    else:
        downtrend = pd.Series(True, index=signal.index)
    return signal.where(narrow & downtrend, other=np.nan)


filter_bear_narrow_lt40.__name__ = "bear_narrow_lt40"


def filter_breadth_lt50(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    """Signal active only when < 50% of stocks are above their 200d MA (narrow breadth)."""
    prices = (1 + cache["returns"]).cumprod()
    ma_200 = prices.rolling(200).mean()
    pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
    mask = pct_above.lt(0.50).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_breadth_lt50.__name__ = "breadth_lt50"


def filter_breadth_strong_55(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    prices = (1 + cache["returns"]).cumprod()
    ma_200 = prices.rolling(200).mean()
    pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
    mask = pct_above.gt(0.55).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_breadth_strong_55.__name__ = "breadth_strong_55"


def filter_vol_contraction_10_60(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    benchmark = cache["benchmark"]
    mask = (
        benchmark.rolling(10)
        .std()
        .lt(benchmark.rolling(60).std())
        .reindex(signal.index)
        .fillna(False)
    )
    return signal.where(mask, other=np.nan)


filter_vol_contraction_10_60.__name__ = "vol_contraction_10_60"


def filter_vol_expansion_10_60(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    benchmark = cache["benchmark"]
    mask = (
        benchmark.rolling(10)
        .std()
        .gt(benchmark.rolling(60).std())
        .reindex(signal.index)
        .fillna(False)
    )
    return signal.where(mask, other=np.nan)


filter_vol_expansion_10_60.__name__ = "vol_expansion_10_60"


def filter_market_trend_down_20_100(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    benchmark = (1 + cache["benchmark"]).cumprod()
    mask = (
        benchmark.rolling(20)
        .mean()
        .le(benchmark.rolling(100).mean())
        .reindex(signal.index)
        .fillna(False)
    )
    return signal.where(mask, other=np.nan)


filter_market_trend_down_20_100.__name__ = "market_trend_down_20_100"


def filter_sector_dislocation_5_q80(
    signal: pd.DataFrame,
    *,
    get_sector_etf_map: SectorEtfMapProvider,
    **cache,
) -> pd.DataFrame:
    returns = cache["_active_returns"]
    factor_returns = cache["factor_returns"]
    sector_returns = _sector_factor_frame(returns, factor_returns, get_sector_etf_map)
    dislocation = (returns.rolling(5).mean() - sector_returns.rolling(5).mean()).abs()
    threshold = dislocation.quantile(0.80, axis=1)
    mask = dislocation.ge(threshold, axis=0)
    return signal.where(mask)


filter_sector_dislocation_5_q80.__name__ = "sector_dislocation_5_q80"


def filter_panic_10d_minus5(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    benchmark = cache["benchmark"]
    mask = benchmark.rolling(10).sum().lt(-0.05).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_panic_10d_minus5.__name__ = "panic_10d_minus5"


# ---------------------------------------------------------------------------
# Scaler factories
# ---------------------------------------------------------------------------


def make_trend_scaler(
    fast: int,
    slow: int,
    *,
    mr_style: bool = True,
    scale_down: float = 0.25,
    name: str | None = None,
) -> ScalerFn:
    def trend_scaler(positions: pd.DataFrame, **cache):
        benchmark = cache["benchmark"]
        equity = (1 + benchmark).cumprod()
        uptrend = (
            equity.rolling(fast)
            .mean()
            .gt(equity.rolling(slow).mean())
            .reindex(positions.index)
            .fillna(False)
        )
        cond = uptrend if mr_style else ~uptrend
        scale = pd.Series(np.where(cond, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    trend_scaler.__name__ = name or f"trend_{fast}_{slow}"
    return trend_scaler


def make_disp_scaler(window: int, low_q: float) -> ScalerFn:
    """Scale down when dispersion is LOW (below low_q percentile). MR/momentum style."""

    def disp_scaler(positions: pd.DataFrame, **cache):
        disp = cache["returns"].std(axis=1).rolling(window).mean()
        low_disp = disp.lt(disp.rolling(252).quantile(low_q)).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(low_disp, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    disp_scaler.__name__ = f"disp_{window}_q{int(low_q * 100)}"
    return disp_scaler


def make_high_disp_scaler(window: int, high_q: float) -> ScalerFn:
    """Scale down when dispersion is HIGH (above high_q percentile). Event/gap style."""

    def high_disp_scaler(positions: pd.DataFrame, **cache):
        r = cache["returns"].dropna(axis=1, how="all")
        disp = r.std(axis=1)
        threshold = disp.rolling(252).quantile(high_q)
        high_disp = disp.gt(threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(high_disp, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    high_disp_scaler.__name__ = f"disp_{window}_q{int(high_q * 100)}"
    return high_disp_scaler


def make_crash_scaler(window: int, threshold: float) -> ScalerFn:
    def crash_scaler(positions: pd.DataFrame, **cache):
        benchmark = cache["benchmark"]
        rebound = (
            benchmark.rolling(window).sum().gt(threshold).reindex(positions.index).fillna(False)
        )
        scale = pd.Series(np.where(rebound, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    crash_scaler.__name__ = f"crash_{window}_{int(threshold * 100)}pct"
    return crash_scaler


def make_vol_scaler_up(fast: int, slow: int) -> ScalerFn:
    """Scale UP in high vol (1.0 when fast_vol > slow_vol, 0.25 otherwise)."""

    def vol_scaler_up(positions: pd.DataFrame, **cache):
        benchmark = cache["benchmark"]
        high_vol = (
            benchmark.rolling(fast)
            .std()
            .gt(benchmark.rolling(slow).std())
            .reindex(positions.index)
            .fillna(False)
        )
        scale = pd.Series(np.where(high_vol, 1.0, 0.25), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    vol_scaler_up.__name__ = f"vol_{fast}_{slow}_up"
    return vol_scaler_up


def make_vol_scaler(fast: int, slow: int, *, scale_down: float = 0.25) -> ScalerFn:
    def vol_scaler(positions: pd.DataFrame, **cache):
        benchmark = cache["benchmark"]
        high_vol = (
            benchmark.rolling(fast)
            .std()
            .gt(benchmark.rolling(slow).std())
            .reindex(positions.index)
            .fillna(False)
        )
        scale = pd.Series(np.where(high_vol, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    vol_scaler.__name__ = f"vol_{fast}_{slow}"
    return vol_scaler


def make_breadth_scaler(threshold: float) -> ScalerFn:
    """Scale down when % stocks above 200d MA is below threshold (momentum-style gate).
    Scale = 0.25 when pct_above_200d < threshold, else 1.0.
    """

    def breadth_scaler(positions: pd.DataFrame, **cache):
        returns = cache.get("returns")
        if returns is None:
            return positions
        prices = (1 + returns).cumprod()
        pct_above = (prices > prices.rolling(200).mean()).mean(axis=1)
        below_thresh = pct_above.lt(threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(below_thresh, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    name_int = str(int(threshold * 100))
    breadth_scaler.__name__ = f"breadth_{name_int}"
    return breadth_scaler


def make_breadth_20_q30_scaler() -> ScalerFn:
    """Scale down when rolling 20d breadth (% positive daily returns) is below 30th pct.
    Breadth is defined as fraction of stocks with positive return on each day.
    """

    def breadth_20_q30(positions: pd.DataFrame, **cache):
        r = cache["returns"].dropna(axis=1, how="all")
        breadth = (r > 0).mean(axis=1)
        threshold = breadth.rolling(252).quantile(0.30)
        low_breadth = breadth.lt(threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(low_breadth, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    breadth_20_q30.__name__ = "breadth_20_q30"
    return breadth_20_q30


# ---------------------------------------------------------------------------
# Sleeve name parser
# ---------------------------------------------------------------------------


def _parse_sleeve_name(name: str) -> tuple[str, int, str, str]:
    signal_and_scaler, cond = name.split("__cond__", 1)
    match = re.fullmatch(
        r"(?P<signal>.+)__r(?P<rebalance>\d+)(?:__(?P<risk_scaling>.+))?", signal_and_scaler
    )
    if match is None:
        raise ValueError(f"Unrecognized sleeve name: {name}")
    return (
        match.group("signal"),
        int(match.group("rebalance")),
        match.group("risk_scaling") or "none",
        cond,
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_sleeve_specs(
    *,
    get_distance_partners: PartnersProvider,
    get_sector_etf_map: SectorEtfMapProvider,
) -> dict[str, SleeveSpec]:
    # (signal_fn, use_factor_model, use_etf_resid)
    signal_factories: dict[str, tuple[SignalFn, bool, bool]] = {
        # Mean Reversion
        "cumret_spread_20_252": (make_cumret_spread(20, 252), False, False),
        "zscore_rev_20_252": (make_zscore_rev(20, 252), False, False),
        # Residual MR
        "etf_factor_resid_mr_5d": (make_residual_mr(5, "etf_factor_resid_mr_5d"), False, True),
        "factor_model_resid_mr_10d": (
            make_residual_mr(10, "factor_model_resid_mr_10d"),
            True,
            False,
        ),
        "resid_zscore_w15_w10": (make_resid_zscore_w15(10), False, True),
        # Distance Pairs
        "dist_mr_k1_z20": (make_dist_mr_k1(20, get_distance_partners), False, False),
        "dist_mr_k1_z60": (make_dist_mr_k1(60, get_distance_partners), False, False),
        "dist_mr_k3_z60": (make_dist_mr(60, get_distance_partners), False, False),
        # Event / Gap Accumulation
        "gap_accum_2d": (make_gap_accum(2), False, False),
        "gap_accum_3d": (make_gap_accum(3), False, False),
        "resid_gap_accum_5d": (make_resid_gap_accum(5), False, True),
        # Momentum
        "sharpe_mom_252d": (make_sharpe_mom(252), False, False),
        "sharpe_mom_120d": (make_sharpe_mom(120), False, False),
        "ma_dist_50_200": (make_ma_dist(50, 200), False, False),
        "ma_dist_20_200": (make_ma_dist(20, 200), False, False),
        "mom_252d": (make_mom(252), False, False),
        # Monotonicity
        "monoton_skip_252d": (make_monoton_skip_252d(), False, False),
        "monoton_w252d": (make_monoton_w252d(), False, False),
        # Residual Momentum
        "resid_mom_252d": (make_resid_mom(252), True, False),
        # Sharpe-Residual Momentum
        "sharpe_resid_mom_252d_skip5": (make_sharpe_resid_mom(252, 5), True, False),
        # Sector-Relative
        "sector_rel_mr_20d": (make_sector_rel_mr(20, get_sector_etf_map), False, False),
        "sector_rel_sharpe_252d_skip5": (
            make_sector_rel_sharpe(252, 5, get_sector_etf_map),
            False,
            True,
        ),
        "sector_rel_mom_120d": (make_sector_rel_mom(120, get_sector_etf_map), False, True),
        # Vol-Trend
        "vol_regime_ret_10_90_r5": (make_vol_regime_ret(10, 90, 5), False, True),
        "ivol_explosion_5_60_p90": (make_ivol_explosion(5, 60, 0.90), False, True),
        "ivol_accel_20_90": (make_ivol_accel(20, 90), False, True),
        "ivol_accel_20_120": (make_ivol_accel(20, 120), False, True),
        # Beta Momentum
        "beta_momentum_60_252d": (make_beta_momentum(60, 252), False, False),
        # Bear Narrow-Breadth
        "vol_accel_20_120d": (make_vol_accel(20, 120), False, False),
        "vol_accel_5_60d": (make_vol_accel(5, 60), False, False),
        "bear_reversal_20d": (make_bear_reversal(20), False, False),
        # Bull Narrow-Breadth
        "low_vol_mom_120d": (make_low_vol_mom(120), False, False),
        # Sector ETF Momentum (narrow-breadth)
        "etf_mom_120d": (make_etf_mom(120, get_sector_etf_map), False, False),
        "etf_sharpe_mom_120d": (make_etf_sharpe_mom(120, get_sector_etf_map), False, False),
        "sector_rel_mom_active_60d": (
            make_sector_rel_mom_active(60, get_sector_etf_map),
            False,
            False,
        ),
    }

    filter_factories: dict[str, Callable[[], FilterFn | None]] = {
        "none": lambda: None,
        # Legacy filters kept for reference
        "residual_dispersion_high_20_q75": lambda: filter_residual_dispersion_high_20_q75,
        "dispersion_high_60_q75": lambda: filter_dispersion_high_60_q75,
        "breadth_weak_40": lambda: filter_breadth_weak_40,
        "breadth_strong_55": lambda: filter_breadth_strong_55,
        "vol_contraction_10_60": lambda: filter_vol_contraction_10_60,
        "vol_expansion_10_60": lambda: filter_vol_expansion_10_60,
        "market_trend_down_20_100": lambda: filter_market_trend_down_20_100,
        "sector_dislocation_5_q80": lambda: _bind_sector_dislocation_filter(get_sector_etf_map),
        "panic_10d_minus5": lambda: filter_panic_10d_minus5,
        "bear_narrow_lt40": lambda: filter_bear_narrow_lt40,
        "breadth_lt40": lambda: filter_breadth_weak_40,
        "breadth_lt50": lambda: filter_breadth_lt50,
        "disp_60_q60": lambda: filter_dispersion_high_60_q60,
    }

    scaler_factories: dict[str, Callable[[], list[ScalerFn]]] = {
        "none": lambda: [],
        # Trend scalers (MR-style: scale down in uptrend)
        "trend_20_100": lambda: [make_trend_scaler(20, 100, mr_style=True)],
        "trend_20_100_mr": lambda: [
            make_trend_scaler(20, 100, mr_style=True, name="trend_20_100_mr")
        ],
        "trend_20_100_h": lambda: [
            make_trend_scaler(20, 100, mr_style=True, scale_down=0.50, name="trend_20_100_h")
        ],
        "trend_50_200": lambda: [make_trend_scaler(50, 200, mr_style=True)],
        "trend_50_200_mr": lambda: [
            make_trend_scaler(50, 200, mr_style=True, name="trend_50_200_mr")
        ],
        # Trend scalers (momentum-style: scale down in downtrend)
        "trend_10_60": lambda: [make_trend_scaler(10, 60, mr_style=False)],
        "trend_50_200_mom": lambda: [
            make_trend_scaler(50, 200, mr_style=False, name="trend_50_200_mom")
        ],
        # Vol scalers
        "vol_10_60": lambda: [make_vol_scaler(10, 60)],
        "vol_20_60": lambda: [make_vol_scaler(20, 60)],
        "vol_20_100": lambda: [make_vol_scaler(20, 100)],
        "vol_20_60_up": lambda: [make_vol_scaler_up(20, 60)],
        # Breadth scalers
        "breadth_40": lambda: [make_breadth_scaler(0.40)],
        "breadth_50": lambda: [make_breadth_scaler(0.50)],
        "breadth_20_q30": lambda: [make_breadth_20_q30_scaler()],
        # Dispersion scalers
        "disp_60_q30": lambda: [make_disp_scaler(60, 0.30)],
        "disp_60_q20": lambda: [make_disp_scaler(60, 0.20)],
        # High-dispersion scaler (event style: scale down when high dispersion)
        "disp_60_q75": lambda: [make_high_disp_scaler(60, 0.75)],
        # Legacy scalers
        "crash_10_5pct": lambda: [make_crash_scaler(10, 0.05)],
    }

    specs: dict[str, SleeveSpec] = {}
    for name in SIGNAL_POOL_SLEEVE_NAMES:
        signal_key, rebalance, scaler_key, filter_key = _parse_sleeve_name(name)
        signal_fn, use_factor_model, use_etf_resid = signal_factories[signal_key]
        conditioning_filter = filter_factories[filter_key]()
        risk_scalers = scaler_factories[scaler_key]()
        needs_resid_cache = filter_key == "residual_dispersion_high_20_q75"
        specs[name] = SleeveSpec(
            name=name,
            signal_fn=signal_fn,
            conditioning_filter=conditioning_filter,
            rebalance_every=rebalance,
            risk_scalers=risk_scalers,
            use_factor_model=use_factor_model,
            use_etf_resid=use_etf_resid,
            needs_resid_cache=needs_resid_cache,
        )

    if len(specs) != 67:
        raise AssertionError(f"Expected 63 sleeves, found {len(specs)}")

    return specs


def _bind_sector_dislocation_filter(get_sector_etf_map: SectorEtfMapProvider) -> FilterFn:
    def sector_dislocation(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        return filter_sector_dislocation_5_q80(
            signal,
            get_sector_etf_map=get_sector_etf_map,
            **cache,
        )

    sector_dislocation.__name__ = "sector_dislocation_5_q80"
    return sector_dislocation
