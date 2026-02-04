from scripts import mc_price, bsm_price

def test_mc_close_to_bsm():
    S = 100.0; K = 100.0; r = 0.01; q = 0.0; sigma = 0.2; T = 0.5
    mc_est, stderr = mc_price(S, K, r, q, sigma, T, option_type="call", n_paths=200000, antithetic=True, control_variate=True, seed=0)
    bs = bsm_price(S, K, r, q, sigma, T, option_type="call")
    assert abs(mc_est - bs) < 3 * stderr + 1e-3