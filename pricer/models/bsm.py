"""
pricer/models/bsm.py
─────────────────────
Black-Scholes-Merton closed-form pricing for European vanilla options.

Responsibilities
- Compute d1 and d2 as private methods (BSM-specific, not shared with other models).
- Return the exact analytical price via the BSM formula.
- Return the five analytical Greeks: delta, gamma, vega, theta, rho.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .base import EuropeanOption


class BlackScholesOption(EuropeanOption):
    """
    European option priced with the Black-Scholes-Merton closed-form formula.

    Inherits all parameters from EuropeanOption.

    Methods
    -------
    price()   -> float              Closed-form option price.
    greeks()  -> dict[str, float]   Analytical delta, gamma, vega, theta, rho.
    """

    # ── BSM private intermediaries ────────────────────────────────────────────

    def _d1(self) -> float:
        """
        BSM d1 term.
        d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
        """
        return (
            np.log(self.S / self.K)
            + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    def _d2(self) -> float:
        """
        BSM d2 term.
        d2 = d1 - sigma * sqrt(T)
        """
        return self._d1() - self.sigma * np.sqrt(self.T)

    # ── price ─────────────────────────────────────────────────────────────────

    def price(self) -> float:
        """
        Return the BSM closed-form price.
        At expiry (T = 0) returns the intrinsic value directly.

        Returns
        -------
        float  Fair value of the option.
        """
        if self.T == 0:
            return float(
                max(0.0, (self.S - self.K) if self.is_call else (self.K - self.S))
            )

        d1, d2   = self._d1(), self._d2()
        disc_r   = np.exp(-self.r * self.T)
        disc_q   = np.exp(-self.q * self.T)

        if self.is_call:
            value = self.S * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        else:
            value = self.K * disc_r * norm.cdf(-d2) - self.S * disc_q * norm.cdf(-d1)

        return float(value)

    # ── analytical Greeks ─────────────────────────────────────────────────────

    def greeks(self) -> dict[str, float | None]:
        """
        Return the five first-order analytical Greeks.
        Returns None for each Greek when T = 0 or sigma = 0 (degenerate case).

        Returns
        -------
        dict with keys: delta, gamma, vega, theta, rho (all float or None).
        """
        _degenerate = {k: None for k in ("delta", "gamma", "vega", "theta", "rho")}
        if self.T == 0 or self.sigma == 0:
            return _degenerate

        d1, d2   = self._d1(), self._d2()
        disc_r   = np.exp(-self.r * self.T)
        disc_q   = np.exp(-self.q * self.T)
        phi_d1   = norm.pdf(d1)
        N_d1     = norm.cdf(d1)
        N_d2     = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)

        # gamma and vega are identical for calls and puts under BSM
        gamma = (disc_q * phi_d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega  = self.S * disc_q * np.sqrt(self.T) * phi_d1

        if self.is_call:
            delta = disc_q * N_d1
            theta = (
                - self.S * self.sigma * disc_q * phi_d1 / (2 * np.sqrt(self.T))
                + self.q * self.S * disc_q * N_d1
                - self.r * self.K * disc_r * N_d2
            )
            rho = self.K * disc_r * self.T * N_d2
        else:
            delta = disc_q * (N_d1 - 1)
            theta = (
                - self.S * self.sigma * disc_q * phi_d1 / (2 * np.sqrt(self.T))
                - self.q * self.S * disc_q * N_neg_d1
                + self.r * self.K * disc_r * N_neg_d2
            )
            rho = -self.K * disc_r * self.T * N_neg_d2

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega":  float(vega),
            "theta": float(theta),
            "rho":   float(rho),
        }