import numpy as np
from typing import Literal, Tuple
from ..utils import seed_rng, payoff_call, payoff_put, validate_positive
from .black_scholes import bsm_price 

OptionType = Literal['call', 'put']

def mc_price (S: float, K: float, r: float, q: float, sigma: float, T: float, 
			  option_type: OptionType = 'call', n_paths: int = 10000, antithetic: bool = True, 
			  control_variate: bool = True, seed: int = None) -> Tuple[float, float]:
	
	# Monte Carlo pricing for European options.
	# Returns (price_estimate, standart_error) using antithetic variates and control variate (if true : use S_T discounted as control variate E)
	validate_positive(S = S, K = K, sigma = sigma, T = T)
	rng = seed_rng(seed)
	dt = T
	
	if antithetic :
	# ensure even number
		if n_paths % 2 != 0:  # modulo to check if it's an odd number
			n_paths +=1
		half = n_paths // 2
		z = rng.standard_normal(size= half)
		z = np.concatenate([z, -z])
	else :
		z = rng.standard_normal(size= n_paths)
	
	ST = S * np.exp( (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z )
	
	if option_type == 'call':
		payoff = payoff_call(ST, K)
	else :
		payoff = payoff_put(ST, K)
	
	discounted = np.exp(-r * T) * payoff


	if control_variate :
		# discounted ST has known expectation = S * exp(-q*T)
		control = np.exp(-r * T) * ST
		control_expectation = S * np.exp(-q * T)
		cov = np.cov(discounted, control,bias= True)[0, 1] # To grab value when stock and option prices move together
		var_control = np.var(control)
		beta = cov / var_control if var_control > 0 else 0.0
		adjusted = discounted - beta * (control - control_expectation)
		estimate = adjusted.mean()
		std_error = adjusted.std(ddof = 1) / np.sqrt(len(adjusted))
	else :
		estimate = discounted.mean()
		std_error = discounted.std(ddof= 1) / np.sqrt(len(discounted)) # ddof : delta degrees of freedom : ensure an unbiased estimate
	
	alanytic = bsm_price(S, K, r, q, sigma, T, option_type)

	return float(estimate), float(std_error)


	

	