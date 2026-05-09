"""
pricer/models/base.py
─────────────────────
Abstract base class shared by every pricing model.

Responsibilities
- Hold and validate the six market/contract parameters (S, K, r, q, sigma, T).
- Provide vectorised payoff helpers (call / put) used by numerical methods.
- Declare the abstract price() interface that every concrete model must implement.

Design note: d1 and d2 are intentionally NOT here.
They are BSM-specific quantities and belong exclusively in bsm.py.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Literal

# ── type alias ────────────────────────────────────────────────────────────────

OptionType = Literal["call", "put"]

# ── module-level helpers ──────────────────────────────────────────────────────

def payoff_call(terminal_prices: np.ndarray, strike: float) -> np.ndarray:
    """Vectorised call payoff: max(S_T - K, 0)."""
    return np.maximum(terminal_prices - strike, 0.0)


def payoff_put(terminal_prices: np.ndarray, strike: float) -> np.ndarray:
    """Vectorised put payoff: max(K - S_T, 0)."""
    return np.maximum(strike - terminal_prices, 0.0)


def validate_positive(**kwargs: float) -> None:
    """Raise ValueError if any named parameter is None or strictly negative."""
    for name, value in kwargs.items():
        if value is None:
            raise ValueError(f"{name} must be provided.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative; got {value}.")


def seed_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy default_rng instance, optionally seeded."""
    return np.random.default_rng(seed)


# ── abstract base class ───────────────────────────────────────────────────────

class EuropeanOption(ABC):
    """
    Abstract base for all European vanilla option pricers.

    Parameters
    ----------
    S           : float       Current spot price of the underlying.
    K           : float       Strike price.
    r           : float       Continuously compounded risk-free rate (annual).
    q           : float       Continuous dividend yield (annual).
    sigma       : float       Volatility (annual).
    T           : float       Time to expiry in years.
    option_type : OptionType  'call' or 'put'.
    """

    def __init__(
        self,
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        option_type: OptionType = "call",
    ) -> None:
        self.S           = S
        self.K           = K
        self.r           = r
        self.q           = q
        self.sigma       = sigma
        self.T           = T
        self.option_type = option_type
        self._validate()

    # ── validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Ensure spot, strike, volatility and time-to-expiry are non-negative."""
        validate_positive(S=self.S, K=self.K, sigma=self.sigma, T=self.T)

    # ── convenience property ──────────────────────────────────────────────────

    @property
    def is_call(self) -> bool:
        """True when the option is a call."""
        return self.option_type == "call"

    # ── payoff ────────────────────────────────────────────────────────────────

    def payoff(self, terminal_prices: np.ndarray) -> np.ndarray:
        """
        Compute the terminal payoff for an array of simulated or grid prices.

        Parameters
        ----------
        terminal_prices : np.ndarray  Prices at expiry S_T.

        Returns
        -------
        np.ndarray  Element-wise payoff values.
        """
        if self.is_call:
            return payoff_call(terminal_prices, self.K)
        return payoff_put(terminal_prices, self.K)

    # ── abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def price(self, *args, **kwargs):
        """Return the fair value of the option. Must be implemented by each model."""
        raise NotImplementedError