"""Shared gamma exposure calculation utilities."""


def row_gross_gex(df, spot, multiplier, gamma_scale):
    """
    Dealer-agnostic (gross) gamma exposure per row.

    Args:
        df: DataFrame with option chain data
        spot: Current underlying price
        multiplier: Contract multiplier (e.g., 100 for SPX)
        gamma_scale: Scaling factor for gamma units (e.g., 0.01)

    Returns:
        Series of gross gamma exposure values per row
    """
    return (
        df["gamma"]
        * df["open_interest"]
        * (spot**2)
        * multiplier
        * gamma_scale
    )


def apply_dealer_sign(value, dealer_short: bool):
    """
    Convert a gross metric into a signed metric under an assumed dealer position.

    Args:
        value: Gross metric value
        dealer_short: If True, assume dealers are short (sign=-1)

    Returns:
        Signed metric value
    """
    sign = -1.0 if dealer_short else 1.0
    return sign * value
