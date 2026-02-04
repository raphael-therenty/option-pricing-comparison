import math
from scripts import bsm_price, bsm_greeks

def test_bsm_call_parity():
    S = 100.0; K = 100.0; r = 0.01; q = 0.0; sigma = 0.2; T = 0.5
    call = bsm_price(S, K, r, q, sigma, T, option_type="call")
    put = bsm_price(S, K, r, q, sigma, T, option_type="put")
    lhs = call - put
    rhs = S * math.exp(-q*T) - K * math.exp(-r*T)
    assert abs(lhs - rhs) < 1e-8