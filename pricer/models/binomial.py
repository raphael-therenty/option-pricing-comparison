"""
pricer/models/binomial.py
──────────────────────────
European option pricing via the Cox-Ross-Rubinstein (CRR) binomial tree.

Responsibilities
- Build a recombining price tree over N time steps.
- Back-propagate option values using risk-neutral probabilities.
- Converge to the BSM price as N increases.
"""

from __future__ import annotations

import numpy as np

from .base import EuropeanOption


class BinomialOption(EuropeanOption):
    """
    European option priced with a CRR recombining binomial tree.

    Inherits all parameters from EuropeanOption.

    Methods
    -------
    price(steps) -> float  Option price from an N-step CRR tree.
    """

    def price(self, steps: int = 100) -> float:
        """
        Price the option using an N-step CRR binomial tree.

        At each step the stock moves up by u = exp(sigma * sqrt(dt))
        or down by d = 1/u. Terminal payoffs are discounted backward
        under the risk-neutral probability p.

        Parameters
        ----------
        steps : int  Number of time steps in the tree (default 100).

        Returns
        -------
        float  Estimated option price at t = 0.
        """
        dt       = self.T / steps
        u        = np.exp(self.sigma * np.sqrt(dt))
        d        = 1.0 / u
        discount = np.exp(-self.r * dt)
        p        = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        # terminal stock prices at all N+1 leaf nodes
        j  = np.arange(steps + 1)
        ST = self.S * (u ** j) * (d ** (steps - j))

        # backward induction from expiry to t = 0
        option_values = self.payoff(ST)
        for i in range(steps - 1, -1, -1):
            option_values = discount * (
                p * option_values[1 : i + 2]
                + (1 - p) * option_values[0 : i + 1]
            )

        return float(option_values[0])