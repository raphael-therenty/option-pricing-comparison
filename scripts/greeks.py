from typing import Callable
import numpy as np

def finite_diff_greeks(price_function: Callable, S: float, bump: float = 1e-4, **kwargs) -> dict :
    # Generic finite-difference greeks with bump and revalue (bump of 1 bp)

    # delta 
    p_plus = price_function(S + bump, **kwargs)
    p_minus = price_function(S - bump, **kwargs)
    delta = (p_plus - p_minus) / (2 * bump)
    
    # gamma
    gamma = (p_plus - 2 * price_function(S, **kwargs) + p_minus) / (bump**2)

    # vega 
    sigma = kwargs.get('sigma')
    if sigma is None:
        vega = None
    else :
        h = max(1e-4, sigma * 1e-3)
        kwargs_sigma_up = dict(kwargs, sigma = sigma + h)
        kwargs_sigma_down = dict(kwargs, sigma = sigma - h)
        vega = (price_function(S, **kwargs_sigma_up) - price_function(S, **kwargs_sigma_down)) / (2 * h)
    
    # theta
    T = kwargs.get('T')
    if T is None or T <= 0 :
        theta = None
    else :
        hT = min(1e-4, T * 1e-4)
        kwargs_T_up = dict(kwargs, T= max(1e-12, T - hT))
        theta = (price_function(S, **kwargs_T_up) - price_function(S, **kwargs)) / (-hT) # per year
    
    # rho
    r = kwargs.get('r')
    if r is None:
        rho = None
    else : 
        hr = 1e-5
        kwargs_r_up = dict(kwargs, r= r+hr)
        kwargs_r_down = dict(kwargs, r= r-hr)
        rho = (price_function(S,**kwargs_r_up) - price_function(S, **kwargs_r_down)) / (2 * hr)
    
    return {'delta': float(delta), 'gamma': float(gamma), 'vega': float(vega), 'theta': float(theta), 'rho': float(rho)}