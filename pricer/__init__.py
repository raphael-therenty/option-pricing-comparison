"""
pricer/__init__.py
───────────────────
Top-level package for the European Option Pricer.

Re-exports the four model classes and bump_greeks so application
code needs only a single import line:
    from pricer import BlackScholesOption, bump_greeks, ...
"""

from .models import (
    EuropeanOption,
    OptionType,
    BlackScholesOption,
    BinomialOption,
    FiniteDifferenceOption,
    MonteCarloOption,
)
from .greeks import bump_greeks

__all__ = [
    "EuropeanOption",
    "OptionType",
    "BlackScholesOption",
    "BinomialOption",
    "FiniteDifferenceOption",
    "MonteCarloOption",
    "bump_greeks",
]