"""
pricer/models/monte_carlo.py
─────────────────────────────
European option pricing via Monte Carlo simulation under the risk-neutral measure.

Responsibilities
- Simulate N log-normal terminal stock prices in one vectorised step.
- Support antithetic variates: pair each draw z with -z to reduce variance.
- Support a control-variate correction using the discounted terminal stock price,
  whose expectation S * exp(-q * T) is known analytically.
- Return both the price estimate and its standard error.
"""

from __future__ import annotations

import numpy as np

from .base import EuropeanOption, seed_rng


class MonteCarloOption(EuropeanOption):
    """
    European option priced by Monte Carlo simulation.

    Inherits all parameters from EuropeanOption.

    Variance-reduction techniques
    ─────────────────────────────
    antithetic      Pair each draw z with -z so errors partially cancel.
    control_variate Use the discounted terminal stock price as a control:
                    its expectation S * exp(-q*T) is known analytically,
                    enabling a regression-based correction of the estimator.

    Methods
    -------
    price(n_paths, antithetic, control_variate, seed) -> (float, float)
        Returns (price_estimate, standard_error).
    """

    def price(
        self,
        n_paths: int = 10_000,
        antithetic: bool = True,
        control_variate: bool = True,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """
        Estimate the option price by averaging discounted payoffs over simulated paths.

        Parameters
        ----------
        n_paths         : int        Number of Monte Carlo paths.
        antithetic      : bool       Mirror each draw with its negative.
        control_variate : bool       Apply regression-based control-variate correction.
        seed            : int|None   RNG seed for reproducibility.

        Returns
        -------
        (estimate, std_error) : tuple[float, float]
        """
        rng = seed_rng(seed)

        # ── draw standard-normal shocks ───────────────────────────────────────
        if antithetic:
            if n_paths % 2 != 0:
                n_paths += 1
            half = n_paths // 2
            z    = rng.standard_normal(size=half)
            z    = np.concatenate([z, -z])
        else:
            z = rng.standard_normal(size=n_paths)

        # ── simulate terminal prices under risk-neutral measure ───────────────
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * self.T
        ST    = self.S * np.exp(drift + self.sigma * np.sqrt(self.T) * z)

        # ── discount payoffs to t = 0 ─────────────────────────────────────────
        discounted = np.exp(-self.r * self.T) * self.payoff(ST)

        # ── control-variate adjustment ────────────────────────────────────────
        if control_variate:
            control      = np.exp(-self.r * self.T) * ST
            control_mean = self.S * np.exp(-self.q * self.T)
            cov_matrix   = np.cov(discounted, control, bias=True)
            var_control  = cov_matrix[1, 1]
            beta         = cov_matrix[0, 1] / var_control if var_control > 0 else 0.0
            adjusted     = discounted - beta * (control - control_mean)
            estimate     = float(adjusted.mean())
            std_error    = float(adjusted.std(ddof=1) / np.sqrt(len(adjusted)))
        else:
            estimate  = float(discounted.mean())
            std_error = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))

        return estimate, std_error