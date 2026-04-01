import numpy as np


def norm_pdf(x):
    """
    Calculate the standard normal probability density function (PDF).

    The standard normal PDF is the bell curve with mean=0 and std=1.

    Args:
        x: Input value(s), can be scalar or numpy array

    Returns:
        PDF value(s) at x: (1/√(2π)) * e^(-x²/2)
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def bs_gamma(s, k, t, sigma, r=0.0, q=0.0):
    """
    Calculate Black-Scholes gamma for European options.

    Gamma measures the rate of change of delta with respect to changes in the
    underlying price. It's the same for both calls and puts.

    Formula: Γ = N'(d1) / (S * sigma * √T)
    where N'(d1) is the standard normal PDF evaluated at d1

    Args:
        s: Spot price of the underlying (S)
        k: Strike price (K)
        t: Time to expiration in years (T)
        sigma: Implied volatility as a decimal (e.g., 0.20 for 20%)
        r: Risk-free interest rate as a decimal (default: 0.0)
        q: Dividend yield as a decimal (default: 0.0)

    Returns:
        Gamma value(s). All inputs can be scalars or numpy arrays.

    Notes:
        - Inputs are floored at a small epsilon (1e-12) to avoid division by zero
        - Gamma is highest when the option is at-the-money (S ≈ K)
        - Gamma approaches zero as the option moves deep in/out of the money
    """
    eps = 1e-12
    s = np.maximum(s, eps)
    k = np.maximum(k, eps)
    t = np.maximum(t, eps)
    sigma = np.maximum(sigma, eps)

    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return norm_pdf(d1) / (s * sigma * np.sqrt(t))
