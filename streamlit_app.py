"""
streamlit_app.py
─────────────────
Interactive Streamlit application for European vanilla option pricing.

Sections
1. Sidebar         — market & contract parameters + per-method controls.
2. Prices table    — side-by-side comparison of all selected methods.
3. Greeks table    — first-order Greeks at the current spot.
4. Greek charts    — Delta, Gamma, Vega, Theta, Rho plotted over a spot range.
5. PnL chart       — long-option PnL profile for a selected method.

Run with:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from pricer import (
    BlackScholesOption,
    BinomialOption,
    FiniteDifferenceOption,
    MonteCarloOption,
    bump_greeks,
)
from pricer.viz import compute_pnl, plot_pnl

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="European Option Pricer", layout="wide")
st.title("European Option Pricer")
st.caption("Created by Raphael Therenty Fradet")

# ── sidebar — market parameters ───────────────────────────────────────────────

st.sidebar.header("Market & Option Parameters")
S           = st.sidebar.number_input("Spot (S)",                value=100.0,  format="%.4f")
K           = st.sidebar.number_input("Strike (K)",              value=100.0,  format="%.4f")
r           = st.sidebar.number_input("Risk-free rate r (annual)", value=0.01, format="%.4f")
q           = st.sidebar.number_input("Dividend yield q (annual)", value=0.0,  format="%.4f")
sigma       = st.sidebar.number_input("Volatility σ (annual)",   value=0.2,    format="%.4f")
T           = st.sidebar.number_input("Time to expiry T (years)", value=0.5,   format="%.4f")
option_type = st.sidebar.selectbox("Option type", ["call", "put"])

st.sidebar.markdown("---")
st.sidebar.header("Method controls")

ALL_METHODS = ["Black-Scholes", "Binomial (CRR)", "Finite Difference (CN)", "Monte Carlo"]
selected    = st.sidebar.multiselect("Methods to display", ALL_METHODS, default=ALL_METHODS)

n_steps     = st.sidebar.slider("Binomial / FD grid steps", 50, 2000, 200, step=50)
n_paths     = st.sidebar.number_input("MC paths (pricing table)", value=50_000, step=1_000)
antithetic  = st.sidebar.checkbox("Antithetic variates (MC)", value=True)
cv          = st.sidebar.checkbox("Control variate (MC)", value=True)
seed        = st.sidebar.number_input("Random seed (MC)", value=42, step=1)

show_mc_greeks  = st.sidebar.checkbox("Include MC in Greek charts (slow)", value=False)
mc_greek_paths  = st.sidebar.number_input("MC paths (Greek charts)", min_value=1_000, max_value=20_000, value=3_000, step=1_000)

# fixed chart range constants (±50% around current spot, 101 points)
S_MIN_MULT    = 0.5
S_MAX_MULT    = 1.5
N_PLOT_POINTS = 101

# ── cached pricers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def price_bsm(S, K, r, q, sigma, T, otype):
    return BlackScholesOption(S, K, r, q, sigma, T, otype).price()

@st.cache_data(show_spinner=False)
def price_binomial(S, K, r, q, sigma, T, otype, steps):
    return BinomialOption(S, K, r, q, sigma, T, otype).price(steps=steps)

@st.cache_data(show_spinner=False)
def price_fd(S, K, r, q, sigma, T, otype, M, N):
    return FiniteDifferenceOption(S, K, r, q, sigma, T, otype).price(s_max_multiplier=3.0, M=M, N=N)

@st.cache_data(show_spinner=False)
def price_mc(S, K, r, q, sigma, T, otype, n_paths, antithetic, cv, seed):
    opt = MonteCarloOption(S, K, r, q, sigma, T, otype)
    return opt.price(n_paths=n_paths, antithetic=antithetic, control_variate=cv, seed=seed)

# ── helper: build a scalar price callable for each method ─────────────────────

def make_price_fn(method: str, steps: int, n_paths: int, antithetic: bool, cv: bool, seed: int):
    """Return a f(S_val, K, r, q, sigma, T, option_type) → float for bump_greeks."""
    if method == "Black-Scholes":
        def fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type):
            return BlackScholesOption(S_val, K, r, q, sigma, T, option_type).price()
    elif method == "Binomial (CRR)":
        def fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type):
            return BinomialOption(S_val, K, r, q, sigma, T, option_type).price(steps=steps)
    elif method == "Finite Difference (CN)":
        def fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type):
            return FiniteDifferenceOption(S_val, K, r, q, sigma, T, option_type).price(
                s_max_multiplier=3.0, M=steps, N=max(10, steps // 2))
    else:  # Monte Carlo
        def fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type):
            opt = MonteCarloOption(S_val, K, r, q, sigma, T, option_type)
            p, _ = opt.price(n_paths=n_paths, antithetic=antithetic, control_variate=cv, seed=seed)
            return p

    # wrap so bump_greeks can call fn(S, sigma=..., T=..., r=..., option_type=...)
    def wrapped(S_val, sigma, T, r, option_type=option_type, **_):
        return fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type)
    return wrapped

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Prices
# ─────────────────────────────────────────────────────────────────────────────

st.header("Prices comparison")
rows = []
if "Black-Scholes" in selected:
    rows.append({"Method": "Black-Scholes", "Price": price_bsm(S, K, r, q, sigma, T, option_type), "Note": "Analytic"})
if "Binomial (CRR)" in selected:
    rows.append({"Method": f"Binomial (steps={n_steps})", "Price": price_binomial(S, K, r, q, sigma, T, option_type, n_steps), "Note": "CRR tree"})
if "Finite Difference (CN)" in selected:
    try:
        p = price_fd(S, K, r, q, sigma, T, option_type, n_steps, max(10, n_steps // 2))
        rows.append({"Method": f"Finite Diff CN (grid={n_steps})", "Price": p, "Note": "Crank-Nicolson"})
    except Exception as e:
        rows.append({"Method": f"Finite Diff CN (grid={n_steps})", "Price": np.nan, "Note": str(e)})
if "Monte Carlo" in selected:
    try:
        p, stderr = price_mc(S, K, r, q, sigma, T, option_type, int(n_paths), antithetic, cv, int(seed))
        rows.append({"Method": f"Monte Carlo (paths={int(n_paths)})", "Price": p, "Note": f"stderr≈{stderr:.4f}"})
    except Exception as e:
        rows.append({"Method": f"Monte Carlo (paths={int(n_paths)})", "Price": np.nan, "Note": str(e)})

df_prices = pd.DataFrame(rows).set_index("Method")
_, col_mid, _ = st.columns([1, 4, 1])
with col_mid:
    st.dataframe(
        df_prices.style
            .format({"Price": "{:.4f}"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ]),
        use_container_width=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Greeks at current spot
# ─────────────────────────────────────────────────────────────────────────────

st.header("Greeks at current spot")
greek_rows = []
GREEK_COLS = ["delta", "gamma", "vega", "theta", "rho"]

if "Black-Scholes" in selected:
    g = BlackScholesOption(S, K, r, q, sigma, T, option_type).greeks()
    greek_rows.append({"Method": "Black-Scholes", **g})

for method in ["Binomial (CRR)", "Finite Difference (CN)", "Monte Carlo"]:
    if method not in selected:
        continue
    mc_p = int(mc_greek_paths) if method == "Monte Carlo" else int(n_paths)
    bump  = 1e-2 if method == "Monte Carlo" else 1e-3
    fn    = make_price_fn(method, n_steps, mc_p, antithetic, cv, int(seed))
    try:
        g = bump_greeks(fn, S, sigma=sigma, T=T, r=r)
    except Exception:
        g = {k: np.nan for k in GREEK_COLS}
    greek_rows.append({"Method": method, **g})

df_greeks = pd.DataFrame(greek_rows).set_index("Method")
_, gcol, _ = st.columns([1, 4, 1])
with gcol:
    st.dataframe(
        df_greeks.style
            .format("{:.4f}")
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ]),
        use_container_width=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Greek charts + PnL
# ─────────────────────────────────────────────────────────────────────────────

st.header("Greeks and PnL")

S_range     = np.linspace(S * S_MIN_MULT, S * S_MAX_MULT, N_PLOT_POINTS)

# Determine which methods to include in the Greek plots
plot_methods = [m for m in selected if m != "Monte Carlo"]
if "Monte Carlo" in selected and show_mc_greeks:
    plot_methods.append("Monte Carlo")

# Accumulate Greek series: accum[greek_name][method_label] = np.ndarray
accum = {g: {} for g in GREEK_COLS}
progress = st.progress(0)
for idx, method in enumerate(plot_methods):
    progress.progress((idx + 1) / max(len(plot_methods), 1))
    mc_p  = int(mc_greek_paths)
    bump  = 1e-2 if method == "Monte Carlo" else 1e-3
    fn    = make_price_fn(method, n_steps, mc_p, antithetic, cv, int(seed))
    label = method

    if method == "Black-Scholes":
        for s_val in S_range:
            g = BlackScholesOption(s_val, K, r, q, sigma, T, option_type).greeks()
            for gn in GREEK_COLS:
                accum[gn].setdefault(label, []).append(g.get(gn, np.nan))
    else:
        for s_val in S_range:
            try:
                g = bump_greeks(fn, s_val, sigma=sigma, T=T, r=r, bump=max(bump, s_val * 1e-4))
            except Exception:
                g = {gn: np.nan for gn in GREEK_COLS}
            for gn in GREEK_COLS:
                accum[gn].setdefault(label, []).append(g.get(gn, np.nan))

    for gn in GREEK_COLS:
        if label in accum[gn]:
            accum[gn][label] = np.array(accum[gn][label])

progress.empty()

# ── render a 3-column grid: row1 = delta/gamma/vega, row2 = theta/rho/pnl ────
figsize = (5.0, 3.2)

def greek_fig(name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    data = accum.get(name, {})
    if not data:
        ax.text(0.5, 0.5, f"No data for {name}", ha="center", va="center", transform=ax.transAxes)
    else:
        for label, vals in data.items():
            ax.plot(S_range, vals, linewidth=1.6, label=label)
        ax.set_title(name.capitalize(), fontsize=11)
        ax.grid(True, linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    return fig

row1 = st.columns(3)
row2 = st.columns(3)

for col, name in zip(row1, ["delta", "gamma", "vega"]):
    with col:
        st.pyplot(greek_fig(name))

for col, name in zip(row2[:2], ["theta", "rho"]):
    with col:
        st.pyplot(greek_fig(name))

# ── PnL chart in the last slot — uses BSM (or first selected method) ─────────
with row2[2]:
    pnl_method = selected[0] if selected else "Black-Scholes"

    def pnl_price_fn(s_val):
        if pnl_method == "Black-Scholes":
            return BlackScholesOption(s_val, K, r, q, sigma, T, option_type).price()
        if pnl_method == "Binomial (CRR)":
            return BinomialOption(s_val, K, r, q, sigma, T, option_type).price(steps=n_steps)
        if pnl_method == "Finite Difference (CN)":
            return FiniteDifferenceOption(s_val, K, r, q, sigma, T, option_type).price(
                s_max_multiplier=3.0, M=n_steps, N=max(10, n_steps // 2))
        opt = MonteCarloOption(s_val, K, r, q, sigma, T, option_type)
        p, _ = opt.price(n_paths=max(2_000, int(n_paths // 10)), antithetic=antithetic,
                         control_variate=cv, seed=int(seed))
        return p

    S_pnl, pnl_vals = compute_pnl(pnl_price_fn, S, s_min_mult=S_MIN_MULT, s_max_mult=S_MAX_MULT, n=200)
    st.pyplot(plot_pnl(S_pnl, pnl_vals, strike=K, title=f"PnL ({pnl_method})"))

st.markdown("---")
st.caption("Tip: disable MC Greeks if charts are slow to render.")
