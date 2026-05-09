"""
pricer/greeks.py
─────────────────
Numerical Greek estimation via symmetric finite-difference bumping.

Responsibilities
- Provide a single bump_greeks() function that works with ANY callable pricer.
- Estimate delta, gamma, vega, theta, rho via symmetric finite differences.
- Keep all bumping logic in one place: binomial, FD and MC all use this.

Design note: BlackScholesOption has exact analytical Greeks in bsm.py.
This module is intended for models without closed-form derivatives.
"""

from __future__ import annotations

from typing import Callable


def bump_greeks(
    price_fn: Callable[..., float],
    S: float,
    sigma: float,
    T: float,
    r: float,
    bump: float = 1e-4,
    **kwargs,
) -> dict[str, float | None]:
    """
    Estimate the five first-order Greeks via symmetric finite differences.

    Parameters
    ----------
    price_fn : Callable  A function f(S, sigma, T, r, **kwargs) -> float.
    S        : float     Current spot price (centre point for bumping).
    sigma    : float     Volatility forwarded to price_fn.
    T        : float     Time to expiry forwarded to price_fn.
    r        : float     Risk-free rate forwarded to price_fn.
    bump     : float     Base bump magnitude; scaled to parameter size internally.
    **kwargs             Any extra keyword arguments forwarded verbatim to price_fn.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho (float or None).
    """

    # ── Delta and Gamma (bump spot) ───────────────────────────────────────────
    hS    = max(bump, abs(S) * 1e-4, 1e-6)
    p0    = price_fn(S,      sigma=sigma, T=T, r=r, **kwargs)
    p_up  = price_fn(S + hS, sigma=sigma, T=T, r=r, **kwargs)
    p_dn  = price_fn(S - hS, sigma=sigma, T=T, r=r, **kwargs)
    delta = (p_up - p_dn)           / (2 * hS)
    gamma = (p_up - 2 * p0 + p_dn) / (hS ** 2)

    # ── Vega (bump volatility) ────────────────────────────────────────────────
    h_sigma  = max(1e-4, sigma * 1e-3)
    p_sig_up = price_fn(S, sigma=sigma + h_sigma, T=T, r=r, **kwargs)
    p_sig_dn = price_fn(S, sigma=sigma - h_sigma, T=T, r=r, **kwargs)
    vega     = (p_sig_up - p_sig_dn) / (2 * h_sigma)

    # ── Theta (bump time; theta = -dV/dT) ────────────────────────────────────
    if T > 0:
        hT    = max(1e-6, T * 1e-4)
        p_T_up = price_fn(S, sigma=sigma, T=T + hT,            r=r, **kwargs)
        p_T_dn = price_fn(S, sigma=sigma, T=max(1e-12, T - hT), r=r, **kwargs)
        theta  = -(p_T_up - p_T_dn) / (2 * hT)
    else:
        theta = None

    # ── Rho (bump risk-free rate) ─────────────────────────────────────────────
    hr    = 1e-5
    p_r_up = price_fn(S, sigma=sigma, T=T, r=r + hr, **kwargs)
    p_r_dn = price_fn(S, sigma=sigma, T=T, r=r - hr, **kwargs)
    rho    = (p_r_up - p_r_dn) / (2 * hr)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "theta": float(theta) if theta is not None else None,
        "rho":   float(rho),
    }