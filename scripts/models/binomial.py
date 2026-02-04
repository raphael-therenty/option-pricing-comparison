import numpy as np
from typing import Literal # To hardcore values (whenever the literal appears, it will always represent the same value). This is useful for initializing variables with known values or defining constants.
from ..utils import payoff_call, payoff_put, validate_positive

OptionType = Literal['call', 'put']

def binomial_price (S: float, K: float, r: float, q: float, sigma: float, T: float, steps: int=100, option_type: OptionType='call') -> float:
    # Cox Ross Rubinstein model

    validate_positive(S=S, K=K, sigma=sigma, T=T)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    discount_factor = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Terminal stock prices
    j = np.arange(start= 0, stop= steps+1) # number of steps
    ST = S * (u ** j) * (d ** (steps - j))
    if option_type == 'call':
        payoff = payoff_call(ST, K)
    else :
        payoff = payoff_put(ST, K)

    # Backward induction 
    for i in range(steps -1, -1, -1) :  # just before maturity (T-1) to 0 | range (i,j) = i + i+1 + ... + j-1
        # at maturity, there is steps + 1 nodes. In each iteration, the array loses one element as the 'Up' and 'Down' child nodes merge into their parent node.
        payoff = discount_factor * (p * payoff[1: i+2] + (1-p) * payoff[0: i+1])
    return float(payoff[0])