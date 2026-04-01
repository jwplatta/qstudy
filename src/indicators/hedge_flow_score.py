"""
Hedge Flow Score (HFS) indicator for intraday decision-making.

Based on the methodology described in GEX_INTRADAY_DECISION_METRIC.md
"""

import numpy as np
import pandas as pd


def calculate_hfs(
    df: pd.DataFrame,
    spot_window_pct: float = 0.01,
    reference_move_pct: float = 0.0025,
    dealer_position: int = -1,
) -> float:
    """
    Calculate normalized Hedge Flow Score for a single option chain snapshot.

    This follows the 5-step procedure from GEX_INTRADAY_DECISION_METRIC.md:
    1. Select local strike window around spot
    2. Compute signed local dealer gamma
    3. Translate gamma into expected hedge flow
    4. Normalize to obtain bounded decision score
    5. Map score to intraday behavior regime

    Args:
        df: DataFrame containing option chain data with columns:
            - strike: strike price
            - gamma: gamma value per contract
            - open_interest: number of contracts
            - underlying_price: current spot price
            - contract_type: 'CALL' or 'PUT' (optional, for debugging)
        spot_window_pct: Window around spot for local gamma calculation
                         (default 1% = ±0.5%). Recommended range: 0.005-0.02
        reference_move_pct: Reference move for HFS calculation
                           (default 0.25%). Typical intraday shock size
        dealer_position: Sign convention for dealer positioning
                        -1 = dealers SHORT gamma (typical for SPX)
                        +1 = dealers LONG gamma (rare, but possible)

    Returns:
        float: Normalized HFS score in range [-1, +1]
               > +0.25: Mean reversion / pinning regime
               < -0.25: Acceleration / breakout regime
               Otherwise: Fragile / chop / tape-driven

    Raises:
        ValueError: If required columns are missing from DataFrame

    Examples:
        >>> import pandas as pd
        >>> df = pd.read_csv("$SPX_exp2025-12-24_2025-12-18_14-30-00.csv")
        >>> hfs = calculate_hfs(df)
        >>> print(f"HFS: {hfs:.3f}")
        >>> if hfs > 0.25:
        ...     print("Regime: Mean reversion / pinning")
        >>> elif hfs < -0.25:
        ...     print("Regime: Acceleration / breakout")
        >>> else:
        ...     print("Regime: Fragile / chop")
    """
    # Validate input
    required_columns = ["strike", "gamma", "open_interest", "underlying_price", "contract_type"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    if df.empty:
        return 0.0

    spot = df["underlying_price"].iloc[0]

    # Step 1: Select local strike window around spot
    # Default: ±0.5% (window_pct=0.01 means ±0.5%)
    window_half = spot_window_pct / 2.0
    lower_bound = spot * (1 - window_half)
    upper_bound = spot * (1 + window_half)

    local_df = df[(df["strike"] >= lower_bound) & (df["strike"] <= upper_bound)].copy()

    if local_df.empty:
        return 0.0

    # Step 2: Compute signed local dealer gamma
    # Formula: Γ_local = Σ (s_i · OI_i · Γ_i · 100 · S)
    # where s_i is the sign based on dealer positioning assumption
    #
    # For SPX, dealers are typically SHORT options (sell to customers), so:
    # - Calls: dealers short calls → when spot rises, they buy (stabilizing)
    # - Puts: dealers short puts → when spot falls, they sell (destabilizing)
    #
    # Net effect depends on whether call or put gamma dominates at current strikes
    MULTIPLIER = 100  # Standard options contract multiplier
    local_df["gamma_exposure"] = local_df["open_interest"] * local_df["gamma"] * MULTIPLIER * spot

    call_gamma = local_df[local_df["contract_type"] == "CALL"]["gamma_exposure"].sum()
    put_gamma = local_df[local_df["contract_type"] == "PUT"]["gamma_exposure"].sum()
    # Net dealer gamma with sign convention (dealers short options)
    gamma_local = dealer_position * (call_gamma - put_gamma)

    # Step 3: Translate gamma into expected hedge flow
    # Formula: HFS = Γ_local · ΔS
    # where ΔS is the reference move (e.g., 0.25% of spot)
    delta_s = spot * reference_move_pct
    hfs = gamma_local * delta_s

    # Step 4: Normalize to obtain bounded decision score
    # Formula: HFS_norm = HFS / (Σ |OI_i · Γ_i| · ΔS)
    # The denominator represents total gamma exposure (unsigned)
    total_gamma = local_df["gamma_exposure"].abs().sum()

    if total_gamma == 0:
        return 0.0

    hfs_norm = hfs / (total_gamma * delta_s)

    # Clamp to [-1, +1] range for safety
    hfs_norm = np.clip(hfs_norm, -1.0, 1.0)

    return hfs_norm


def interpret_hfs(hfs_score: float) -> dict:
    """
    Interpret HFS score and return regime classification.

    Args:
        hfs_score: Normalized HFS score in range [-1, +1]

    Returns:
        dict with keys:
            - regime: str describing the market regime
            - description: str with interpretation
            - threshold: str indicating which threshold was crossed

    Examples:
        >>> result = interpret_hfs(0.35)
        >>> print(result['regime'])
        'Mean Reversion / Pinning'
        >>> print(result['description'])
        'Dealer hedging flows dampen price moves. Expect mean reversion.'
    """
    if hfs_score > 0.25:
        return {
            "regime": "Mean Reversion / Pinning",
            "description": "Dealer hedging flows dampen price moves. Expect mean reversion.",
            "threshold": ">+0.25",
            "score": hfs_score,
        }
    elif hfs_score < -0.25:
        return {
            "regime": "Acceleration / Breakout",
            "description": "Dealer hedging amplifies price moves. Breakout risk elevated.",
            "threshold": "<-0.25",
            "score": hfs_score,
        }
    else:
        return {
            "regime": "Fragile / Chop / Tape-Driven",
            "description": "Neutral dealer flow. Price action driven by orderflow and news.",
            "threshold": "[-0.25, +0.25]",
            "score": hfs_score,
        }
