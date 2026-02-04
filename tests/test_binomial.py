from scripts import bsm_price, binomial_price

def test_binomial_converges_to_bsm():
    S = 100.0; K = 100.0; r = 0.01; q = 0.0; sigma = 0.2; T = 0.5
    bs = bsm_price(S, K, r, q, sigma, T, option_type="call")
    binom = binomial_price(S, K, r, q, sigma, T, steps=2000, option_type="call")
    assert abs(bs - binom) < 5e-3