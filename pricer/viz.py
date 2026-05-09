"""
pricer/viz.py
──────────────
Visualisation utilities for the Streamlit app.

Responsibilities
- Compute the PnL profile of a long option position across a spot range.
- Produce a ready-to-render Matplotlib figure for st.pyplot().

Separation of concerns: calculation (compute_pnl) and rendering (plot_pnl)
are two distinct functions so each can be tested or reused independently.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def compute_pnl(
    price_fn: Callable[..., float],
    S0: float,
    s_min_mult: float = 0.5,
    s_max_mult: float = 1.5,
    n: int = 100,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the PnL of a long-one-option position across a spot range.

    PnL(S) = option_price(S) - option_price(S0)

    Parameters
    ----------
    price_fn   : Callable  Function f(S, **kwargs) -> float.
    S0         : float     Reference spot (the purchase price point).
    s_min_mult : float     Lower bound = S0 * s_min_mult.
    s_max_mult : float     Upper bound = S0 * s_max_mult.
    n          : int       Number of evaluation points.
    **kwargs               Forwarded verbatim to price_fn.

    Returns
    -------
    (S_range, pnl) : tuple[np.ndarray, np.ndarray]
    """
    S_range = np.linspace(S0 * s_min_mult, S0 * s_max_mult, n)
    p0      = price_fn(S0, **kwargs)
    prices  = np.array([price_fn(s, **kwargs) for s in S_range])
    return S_range, prices - p0


def plot_pnl(
    S_range: np.ndarray,
    pnl: np.ndarray,
    strike: float | None = None,
    title: str = "PnL",
) -> plt.Figure:
    """
    Build a Matplotlib figure of the PnL profile.

    Parameters
    ----------
    S_range : np.ndarray   Spot prices on the x-axis.
    pnl     : np.ndarray   Corresponding PnL values.
    strike  : float|None   If provided, draw a vertical dashed line at K.
    title   : str          Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(S_range, pnl, linewidth=1.8)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    if strike is not None:
        ax.axvline(strike, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Underlying price")
    ax.set_ylabel("PnL")
    ax.set_title(title, fontsize=11)
    ax.grid(True, linewidth=0.5)
    plt.tight_layout()
    return fig