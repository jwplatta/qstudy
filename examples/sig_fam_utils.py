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


TOP5_BY_SIG_FAM_SLEEVE_NAMES = [
    "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "factor_model_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "factor_model_resid_mr_2d__r10__trend_50_200__cond__dispersion_high_60_q75",
    "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "etf_factor_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "dist_mr_k3_z60__r10__cond__none",
    "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
    "dist_mr_k3_z60__r5__cond__vol_expansion_10_60",
    "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
    "dist_mr_k3_z10__r10__cond__vol_contraction_10_60",
    "mr_5d__r10__trend_20_100__cond__breadth_weak_40",
    "mr_5d__r10__trend_50_200__cond__breadth_weak_40",
    "mr_5d__r10__trend_20_100_h__cond__breadth_weak_40",
    "etf_factor_resid_mr_5d__r10__trend_20_100__cond__market_trend_down_20_100",
    "etf_factor_resid_mr_5d__r10__trend_20_100_h__cond__market_trend_down_20_100",
    "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "sector_rel_zscore_5_60__r10__trend_20_100__cond__vol_contraction_10_60",
    "sector_rel_zscore_5_60__r10__trend_20_100_h__cond__vol_contraction_10_60",
    "monoton_120d__r21__disp_60_q30__cond__none",
    "monoton_120d__r21__crash_10_5pct__cond__market_trend_down_20_100",
    "monoton_120d__r21__vol_20_60__cond__panic_10d_minus5",
    "monoton_120d__r21__disp_60_q20__cond__none",
    "monoton_120d__r21__vol_20_60__cond__none",
    "sector_rel_zscore_5_60__r10__trend_20_100__cond__breadth_weak_40",
    "sector_rel_zscore_5_60__r10__trend_20_100_h__cond__breadth_weak_40",
    "resid_gap_reversion__r10__trend_50_200__cond__sector_dislocation_5_q80",
    "resid_gap_reversion__r10__trend_50_200__cond__vol_contraction_10_60",
    "resid_gap_reversion__r10__trend_50_200__cond__breadth_strong_55",
    "resid_gap_reversion__r10__none__cond__none",
    "resid_gap_reversion__r10__trend_50_200__cond__none",
    "dist_mr_k3_z10__r10__cond__panic_10d_minus5",
]


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


def make_monoton_120d() -> SignalFn:
    def monoton_120d(**cache):
        returns = cache["_active_returns"]
        mean_120 = returns.rolling(120).mean()
        same_sign = (returns.gt(0) == mean_120.gt(0)).astype(float)
        return same_sign.rolling(120).mean() * mean_120.abs()

    monoton_120d.__name__ = "monoton_120d"
    return monoton_120d


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


def make_residual_mr(window: int, name: str) -> SignalFn:
    def residual_mr(**cache):
        return -cache["residual_returns"].rolling(window).mean()

    residual_mr.__name__ = name
    return residual_mr


def make_dist_mr(zw: int, get_distance_partners: PartnersProvider) -> SignalFn:
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


def make_resid_gap_reversion() -> SignalFn:
    def resid_gap_reversion(**cache):
        return -cache["residual_returns"].shift(1)

    resid_gap_reversion.__name__ = "resid_gap_reversion"
    return resid_gap_reversion


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


def filter_breadth_weak_40(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    prices = (1 + cache["returns"]).cumprod()
    ma_200 = prices.rolling(200).mean()
    pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
    mask = pct_above.lt(0.40).reindex(signal.index).fillna(False)
    return signal.where(mask, other=np.nan)


filter_breadth_weak_40.__name__ = "breadth_weak_40"


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
    def disp_scaler(positions: pd.DataFrame, **cache):
        disp = cache["returns"].std(axis=1).rolling(window).mean()
        low_disp = disp.lt(disp.rolling(252).quantile(low_q)).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(low_disp, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    disp_scaler.__name__ = f"disp_{window}_q{int(low_q * 100)}"
    return disp_scaler


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


def build_top5_by_sig_fam_sleeve_specs(
    *,
    get_distance_partners: PartnersProvider,
    get_sector_etf_map: SectorEtfMapProvider,
) -> dict[str, SleeveSpec]:
    signal_factories: dict[str, tuple[SignalFn, bool, bool]] = {
        "zscore_rev_5_60": (make_zscore_rev(5, 60), False, False),
        "factor_model_resid_mr_5d": (make_residual_mr(5, "factor_model_resid_mr_5d"), True, False),
        "factor_model_resid_mr_2d": (make_residual_mr(2, "factor_model_resid_mr_2d"), True, False),
        "etf_factor_resid_mr_5d": (make_residual_mr(5, "etf_factor_resid_mr_5d"), False, True),
        "dist_mr_k3_z10": (make_dist_mr(10, get_distance_partners), False, False),
        "dist_mr_k3_z20": (make_dist_mr(20, get_distance_partners), False, False),
        "dist_mr_k3_z60": (make_dist_mr(60, get_distance_partners), False, False),
        "mr_5d": (make_mr(5), False, False),
        "sector_rel_mr_5d": (make_sector_rel_mr(5, get_sector_etf_map), False, False),
        "sector_rel_zscore_5_60": (
            make_sector_rel_zscore_rev(5, 60, get_sector_etf_map),
            False,
            False,
        ),
        "resid_gap_reversion": (make_resid_gap_reversion(), False, True),
        "monoton_120d": (make_monoton_120d(), False, False),
    }

    filter_factories: dict[str, Callable[[], FilterFn | None]] = {
        "none": lambda: None,
        "residual_dispersion_high_20_q75": lambda: filter_residual_dispersion_high_20_q75,
        "dispersion_high_60_q75": lambda: filter_dispersion_high_60_q75,
        "breadth_weak_40": lambda: filter_breadth_weak_40,
        "breadth_strong_55": lambda: filter_breadth_strong_55,
        "vol_contraction_10_60": lambda: filter_vol_contraction_10_60,
        "vol_expansion_10_60": lambda: filter_vol_expansion_10_60,
        "market_trend_down_20_100": lambda: filter_market_trend_down_20_100,
        "sector_dislocation_5_q80": lambda: _bind_sector_dislocation_filter(get_sector_etf_map),
        "panic_10d_minus5": lambda: filter_panic_10d_minus5,
    }

    scaler_factories: dict[str, Callable[[], list[ScalerFn]]] = {
        "none": lambda: [],
        "trend_20_100": lambda: [make_trend_scaler(20, 100, mr_style=True)],
        # "_h" is treated as the half-strength trend overlay.
        "trend_20_100_h": lambda: [
            make_trend_scaler(
                20,
                100,
                mr_style=True,
                scale_down=0.50,
                name="trend_20_100_h",
            )
        ],
        "trend_50_200": lambda: [make_trend_scaler(50, 200, mr_style=True)],
        "disp_60_q30": lambda: [make_disp_scaler(60, 0.30)],
        "disp_60_q20": lambda: [make_disp_scaler(60, 0.20)],
        "crash_10_5pct": lambda: [make_crash_scaler(10, 0.05)],
        "vol_20_60": lambda: [make_vol_scaler(20, 60)],
    }

    specs: dict[str, SleeveSpec] = {}
    for name in TOP5_BY_SIG_FAM_SLEEVE_NAMES:
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

    if len(specs) != 31:
        raise AssertionError(f"Expected 31 sleeves, found {len(specs)}")

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
