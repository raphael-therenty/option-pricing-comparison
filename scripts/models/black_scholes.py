import numpy as np
from scipy.stats import norm
from ..utils import validate_positive

def d1d2 (S, K, r, q, sigma, T) :
	d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	return d1, d2

def bsm_price (S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = 'call') -> float:
	# Black-Scholes-Merton closed-form price for Europen options.
	validate_positive(S= S, K= K, sigma= sigma, T= T)
	if T == 0 :
		return max(0.0, (S-K) if option_type == 'call' else (K-S))
	d1,d2 = d1d2(S, K, r, q, sigma, T)
	if option_type == 'call' :
		price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
	else :
		price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
	return float(price)

def bsm_greeks (S: float, K: float, r: float, q: float, sigma : float, T: float, option_type: str = 'call') -> dict :
	validate_positive(S= S, K= K, sigma= sigma, T= T)
	
	# fallback to numerical or trivial
	if T == 0 or sigma == 0 :
		return {'delta': None, 'gamma': None, 'vega': None, 'theta': None, 'rho': None}
	d1, d2 = d1d2(S, K, r, q, sigma, T)
	density_d1 = norm.pdf(d1) # N' -> derivate of an integral => density function
	N_d1 = norm.cdf(d1)
	N_d2 = norm.cdf(d2)
	N_neg_d1 = norm.cdf(-d1)
	N_neg_d2 = norm.cdf(-d2)

	if option_type == 'call' :
		delta = np.exp(-q * T) * N_d1
		gamma = (np.exp(-q * T) * density_d1) / (S * sigma * np.sqrt(T))
		vega = S * np.exp(-q * T) * np.sqrt(T) * density_d1 
		theta = (-S * sigma * np.exp(-q * T) * density_d1  / (2 * np.sqrt(T)) 
				 +q * S * np.exp(-q * T) * N_d1 
				 -r * K * np.exp(-r * T) * N_d2
		)
		rho = K * np.exp(-r * T) * T * N_d2
		
	
	else :
		delta = np.exp(-q * T) * (N_d1 - 1)
		gamma = (np.exp(-q *T) * density_d1) / (S * sigma * np.sqrt(T))
		vega = S * np.exp(-q * T) * np.sqrt(T) * density_d1
		theta = ((-S * sigma * np.exp(-q * T) * density_d1) / (2 * np.sqrt(T))
		   		 -q * S * np.exp(-q * T) * N_neg_d1
				 +r * K * np.exp(-r * T) * N_neg_d2
		   )
		rho = -K * np.exp(-r * T) * T * N_neg_d2
	
	return {
		'delta': float(delta),
		'gamma': float(gamma),
		'vega': float(vega),
		'theta': float(theta), # per year
		'rho': float(rho)
	}
