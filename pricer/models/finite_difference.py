"""
pricer/models/finite_difference.py
────────────────────────────────────
European option pricing via the Crank-Nicolson finite-difference PDE solver.

Responsibilities
- Discretise the Black-Scholes PDE on a (S, t) grid.
- Apply Crank-Nicolson time-stepping (second-order accurate, unconditionally stable).
- Solve the resulting banded linear system at each time step.
- Interpolate the grid solution back to the current spot price.

Design note: Greek bumping logic is NOT here.
It lives in pricer/greeks.py and works with any model via _price_at_spot().
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from .base import EuropeanOption


class FiniteDifferenceOption(EuropeanOption):
    """
    European option priced by solving the Black-Scholes PDE with the
    Crank-Nicolson implicit finite-difference scheme.

    Inherits all parameters from EuropeanOption.

    Methods
    -------
    price(s_max_multiplier, M, N) -> float  PDE-grid option price.
    _price_at_spot(S_val, M, N)   -> float  Internal helper for Greek bumping.
    """

    def price(
        self,
        s_max_multiplier: float = 3.0,
        M: int = 400,
        N: int = 400,
    ) -> float:
        """
        Solve the BSM PDE backward in time on an (M+1) x (N+1) grid.

        The stock-price axis runs from 0 to S_max = max(S, K) * s_max_multiplier.
        Crank-Nicolson blends explicit and implicit Euler equally,
        giving O(dt^2, dS^2) convergence without stability constraints.

        Parameters
        ----------
        s_max_multiplier : float  Upper boundary = max(S, K) * this value.
        M                : int    Number of spatial (price) grid steps.
        N                : int    Number of temporal (time) grid steps.

        Returns
        -------
        float  Option price at (S, t=0) via linear interpolation on the grid.
        """
        S_max  = max(self.S, self.K) * s_max_multiplier
        dt     = self.T / N
        S_grid = np.linspace(0, S_max, M + 1)
        grid   = np.zeros((M + 1, N + 1))

        # ── terminal condition (payoff at expiry) ─────────────────────────────
        grid[:, -1] = self.payoff(S_grid)

        # ── boundary conditions for all t ─────────────────────────────────────
        tau = np.linspace(self.T, 0, N + 1)   # time remaining at each column
        if self.is_call:
            grid[-1, :] = S_max - self.K * np.exp(-self.r * tau)
            grid[0,  :] = 0.0
        else:
            grid[-1, :] = 0.0
            grid[0,  :] = self.K * np.exp(-self.r * tau)

        # ── Crank-Nicolson coefficients for interior nodes ────────────────────
        j = np.arange(1, M)
        a = 0.25 * dt * (self.sigma ** 2 * j ** 2 - (self.r - self.q) * j)
        b = -0.5 * dt * (self.sigma ** 2 * j ** 2 + self.r)
        c = 0.25 * dt * (self.sigma ** 2 * j ** 2 + (self.r - self.q) * j)

        # LHS matrix A (implicit part)
        A_diag = 1 - b
        A_sub  = -a
        A_sup  = -c

        # RHS matrix B (explicit part applied to previous time step)
        B_diag = 1 + b
        B_sub  =  a
        B_sup  =  c

        # ── backward time-march ───────────────────────────────────────────────
        for n in reversed(range(N)):
            rhs = (
                B_sub  * grid[0:-2, n + 1]
                + B_diag * grid[1:-1, n + 1]
                + B_sup  * grid[2:,   n + 1]
            )
            rhs[0]  += B_sub[0]  * grid[0,  n + 1] + A_sub[0]  * grid[0,  n]
            rhs[-1] += B_sup[-1] * grid[-1, n + 1] + A_sup[-1] * grid[-1, n]

            ab         = np.zeros((3, M - 1))
            ab[0, 1:]  = A_sup[:-1]
            ab[1, :]   = A_diag
            ab[2, :-1] = A_sub[1:]

            grid[1:-1, n] = solve_banded((1, 1), ab, rhs)

        return float(np.interp(self.S, S_grid, grid[:, 0]))

    # ── internal helper ───────────────────────────────────────────────────────

    def _price_at_spot(self, S_val: float, M: int = 400, N: int = 400) -> float:
        """
        Re-price with a temporarily modified spot S_val.
        Used by bump_greeks() in pricer/greeks.py.
        Restores self.S to its original value via finally block.
        """
        original_S = self.S
        self.S = S_val
        try:
            return self.price(s_max_multiplier=3.0, M=M, N=N)
        finally:
            self.S = original_S