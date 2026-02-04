import numpy as np
from scipy.linalg import solve_banded
from typing import Literal 
from ..utils import payoff_call, payoff_put, validate_positive

OptionType = Literal['call', 'put']

def fd_price_cn(S: float, K: float, r: float, q: float, sigma: float, T: float, 
                s_max_multiplier: float = 3.0, M : int= 400, N: int = 400, 
                option_type: OptionType='call') -> float:
    
    # Crank-Nicolson finite difference for European options (S_max, M price grid points, N time steps) with interpolation

    validate_positive(S=S, K= K, sigma= sigma, T= T)
    S_max = max(S,K) * s_max_multiplier
    ds = S_max / M
    dt = T / N

    # grid: price [0, ..., S_max]
    grid = np.zeros((M+1, N+1))
    S_grid = np.linspace(0, S_max, M+1)  # split the interval with fixed discrepancies
    
    # terminal payoff
    if option_type == 'call':
        grid[:, -1] = payoff_call(S_grid, K) # start with last column of the grid bc the payoff is known.
    else :
        grid[:, -1] = payoff_put(S_grid, K)

    # boundary condition for all t
    tau = np.linspace(T, 0, N+1) # time remaining at each step
    if option_type == 'call':
        grid[-1, :] = S_max - K * np.exp(-r * tau) # deep ITM call (S = S_max)
        grid[0, :] = 0.0    # if S = 0
    else:
        grid[-1, :] = 0.0    # deep OTM put (S = S_max)
        grid[0, :] = K * np.exp(-r * tau)  # deep ITM put (S = 0)

    # coefficients for CN
    j = np.arange(1, M)

    a = 0.25 * dt * (sigma**2 * j**2 - (r - q) * j)
    b = -0.5 * dt * (sigma**2 * j**2 + r)
    c = 0.25 * dt * (sigma**2 * j**2 + (r - q) * j)

    # emplicit 
    A_diag = 1-b 
    A_sub = -a
    A_sup = -c

    # explicit
    B_diag = 1 + b
    B_sub = a
    B_sup = c

    # time-stepping
    for n in reversed(range(N)) :
        # righ hand side B * V(n+1)				  n+1 => prices at the future step 
        rhs = (B_sub * grid[0:-2, n+1] 			# doneighbor below price | 0 to M-2
               + B_diag * grid[1:-1, n+1] 		# neighbor same price    | 1 to M-1
               + B_sup * grid[2:, n+1])			# neighbor above price   | 2 to M

        # boundaries with future values
        rhs[0] += B_sub[0] * grid[0, n+1] + A_sub[0] * grid[0, n]
        rhs[-1] += B_sup[-1] * grid[-1, n+1] + A_sup[-1] * grid[-1, n]

        # solve A*x = rhs using banded representaiton
        ab = np.zeros((3, M-1))
        ab[0, 1:] = A_sup[:-1]   # upper diag offset by 1
        ab[1, :] = A_diag       # main diag
        ab[2, :-1] = A_sub[1:] # lower diag offset by 1
        x = solve_banded((1,1), ab, rhs)
        grid[1:-1, n] = x
    
    price = np.interp(S, S_grid, grid[:,0])
    return float(price)
 
