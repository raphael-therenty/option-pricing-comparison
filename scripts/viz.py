import numpy as np
import matplotlib.pyplot as plt

def pnl_from_pricing_method(pricer, S0, K, r, q, sigma, T, 
                            method_kwargs= None, s_min_mult= 0.5, s_max_mult= 1.5, n = 100):
    
    if method_kwargs is None :
        method_kwargs = {}
    
    S_range = np.linspace(S0 * s_min_mult, S0 * s_max_mult, n)
    prices = []
    
    for s in S_range :
        price = pricer(s, K= K, r= r, q= q, sigma= sigma, T= T, **method_kwargs)
        prices.append(price)
        
        # PnL for long one option = price - (price at S0)
        p0 = pricer(S0, K= K, r= r, q= q, sigma= sigma, T= T, **method_kwargs)
    pnl = np.array(prices) - p0
    return S_range, pnl

def plot_payoff(S_range, payoff_values, title='Payoff / PnL', show_strike= None) :
    fig, ax = plt.subplots(figsize= (7,4))
    ax.plot(S_range, payoff_values)
    ax.set_xlabel('Underlying Price')
    ax.set_ylabel('Payoff / PnL')
    ax.set_title(title)
    ax.grid(True)
    
    if show_strike is not None:
        ax.axvline(show_strike, color= 'k', linestyle= '--', linewidth= 0.6)
    return fig


