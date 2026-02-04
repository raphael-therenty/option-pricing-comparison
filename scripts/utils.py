from typing import Callable # To clarify code and check Type of variable (->)
import numpy as np

def payoff_call(s: np.ndarray, k: float) -> np.ndarray :
    return np.maximum(s - k, 0.0)

def payoff_put(s : np.ndarray, k: float) -> np.ndarray :
    return np.maximum(k - s, 0.0)

def validate_positive(**kwargs) : 
    for name, value in kwargs.items():
        if value is None :
            raise ValueError(f'{name} must be provided')
        if value < 0 :
            raise ValueError(f'{name} must be non-negative: got value {value}')

def seed_rng(seed: int = None):
    return np.random.default_rng(seed)
