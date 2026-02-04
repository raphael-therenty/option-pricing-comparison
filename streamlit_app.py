import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `from src...` works when Streamlit runs from app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import library exports
from scripts import (
    bsm_price,
    bsm_greeks,
    binomial_price,
    fd_price_cn,
    mc_price,
    finite_diff_greeks,
)
from scripts.viz import pnl_from_pricing_method

st.set_page_config(page_title="European Option Pricer", layout="wide")
st.title("European Option Pricing: Overview & Comparison")
st.write("Created by Raphael Therenty")

# ------------------------
# Sidebar - market params
# ------------------------
st.sidebar.header("Market & Option Parameters")
S = st.sidebar.number_input("Spot (S)", value=100.0, format="%.4f")
K = st.sidebar.number_input("Strike (K)", value=100.0, format="%.4f")
r = st.sidebar.number_input("Risk-free rate r (annual)", value=0.01, format="%.4f")
q = st.sidebar.number_input("Dividend yield q (annual)", value=0.0, format="%.4f")
sigma = st.sidebar.number_input("Volatility sigma (annual)", value=0.2, format="%.4f")
T = st.sidebar.number_input("Time to expiry T (years)", value=0.5, format="%.4f")
option_type = st.sidebar.selectbox("Option type", ["call", "put"])

st.sidebar.markdown("---")
st.sidebar.header("Method controls")
methods_available = ["Black-Scholes", "Binomial (CRR)", "Finite Difference (CN)", "Monte Carlo"]
selected_methods = st.sidebar.multiselect("Methods to include", options=methods_available, default=methods_available)

n_steps = st.sidebar.slider("Binomial / FD steps (grid size)", min_value=50, max_value=2000, value=200, step=50)
n_paths = st.sidebar.number_input("MC paths (table)", value=50000, step=1000)
plot_mc_with_low_paths = st.sidebar.checkbox("Include MC in greek charts (use small paths)", value=False)
mc_greek_paths = st.sidebar.number_input("MC paths (greeks plots)", min_value=1000, max_value=20000, value=3000, step=1000)
antithetic = st.sidebar.checkbox("Use antithetic (MC)", value=True)
control_variate = st.sidebar.checkbox("Use control variate (MC)", value=True)
seed = st.sidebar.number_input("Random seed (MC)", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Visualization")
s_min_mult = st.sidebar.slider("S range lower multiplier", 0.2, 1.0, 0.5)
s_max_mult = st.sidebar.slider("S range upper multiplier", 1.0, 2.0, 1.5)
n_plot_points = st.sidebar.slider("Points for greek plots", 25, 201, 101, step=2)

# ------------------------
# Cached helpers
# ------------------------
@st.cache_data(show_spinner=False)
def get_price_bsm(S_, K_, r_, q_, sigma_, T_, option_type_):
    return bsm_price(S_, K_, r_, q_, sigma_, T_, option_type=option_type_)

@st.cache_data(show_spinner=False)
def get_price_binomial(S_, K_, r_, q_, sigma_, T_, steps_, option_type_):
    return binomial_price(S_, K_, r_, q_, sigma_, T_, steps=steps_, option_type=option_type_)

@st.cache_data(show_spinner=False)
def get_price_fd(S_, K_, r_, q_, sigma_, T_, M_, N_, option_type_):
    return fd_price_cn(S_, K_, r_, q_, sigma_, T_, s_max_multiplier=3.0, M=M_, N=N_, option_type=option_type_)

@st.cache_data(show_spinner=False)
def get_price_mc(S_, K_, r_, q_, sigma_, T_, option_type_, n_paths_, antithetic_, control_variate_, seed_):
    p, stderr = mc_price(S_, K_, r_, q_, sigma_, T_, option_type=option_type_, n_paths=n_paths_,
                         antithetic=antithetic_, control_variate=control_variate_, seed=seed_)
    return p, stderr

# ------------------------
# Price table 
# ------------------------
st.header("Prices comparison")

prices = []
if "Black-Scholes" in selected_methods:
    p = get_price_bsm(S, K, r, q, sigma, T, option_type)
    prices.append({"Method": "Black-Scholes", "Price": p, "Note": "Analytic"})

if "Binomial (CRR)" in selected_methods:
    p = get_price_binomial(S, K, r, q, sigma, T, n_steps, option_type)
    prices.append({"Method": f"Binomial (steps={n_steps})", "Price": p, "Note": "CRR tree"})

if "Finite Difference (CN)" in selected_methods:
    try:
        p = get_price_fd(S, K, r, q, sigma, T, M_=n_steps, N_=max(10, n_steps//2), option_type_=option_type)
        prices.append({"Method": f"FiniteDiff CN (grid={n_steps})", "Price": p, "Note": "Crank-Nicolson"})
    except Exception as e:
        prices.append({"Method": f"FiniteDiff CN (grid={n_steps})", "Price": np.nan, "Note": f"Error: {e}"})

if "Monte Carlo" in selected_methods:
    try:
        p, stderr = get_price_mc(S, K, r, q, sigma, T, option_type, n_paths, antithetic, control_variate, seed)
        prices.append({"Method": f"Monte Carlo (paths={n_paths})", "Price": p, "Note": f"stderr≈{stderr:.4f}"})
    except Exception as e:
        prices.append({"Method": f"Monte Carlo (paths={n_paths})", "Price": np.nan, "Note": f"Error: {e}"})

prices_df = pd.DataFrame(prices).set_index("Method")

# Styling: center headers and cells, 4 decimals
styler = prices_df.style.format({"Price": "{:.4f}", "Note": "{}"}) \
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
    ])

# Put centered by placing in middle column of a wider layout
c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    st.dataframe(styler, use_container_width=True)

st.markdown("---")

# ------------------------
# Greeks table at current S
# ------------------------
st.header("Greeks at current spot (S)")
greeks_rows = []

# analytic BSM greeks
if "Black-Scholes" in selected_methods:
    try:
        g = bsm_greeks(S, K, r, q, sigma, T, option_type=option_type)
        greeks_rows.append({"Method": "Black-Scholes", **g})
    except Exception:
        greeks_rows.append({"Method": "Black-Scholes", "delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan, "rho": np.nan})

# numeric greeks helper builders
def build_price_func_for_method(method_name):
    if method_name == "Binomial":
        def price_fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type):
            return binomial_price(S_val, K, r, q, sigma, T, steps=n_steps, option_type=option_type)
        return price_fn
    if method_name == "FiniteDiff":
        def price_fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type):
            return fd_price_cn(S_val, K, r, q, sigma, T, s_max_multiplier=3.0, M=n_steps, N=max(10, n_steps//2), option_type=option_type)
        return price_fn
    if method_name == "MonteCarlo":
        def price_fn(S_val, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type):
            p, _ = mc_price(S_val, K, r, q, sigma, T, option_type=option_type, n_paths=max(1000, int(mc_greek_paths)), antithetic=antithetic, control_variate=control_variate, seed=seed)
            return p
        return price_fn
    return None

# compute numeric greeks for chosen methods
if "Binomial (CRR)" in selected_methods:
    price_fn = build_price_func_for_method("Binomial")
    try:
        gnum = finite_diff_greeks(price_fn, S, bump=1e-3, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type)
    except Exception:
        gnum = {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan, "rho": np.nan}
    greeks_rows.append({"Method": f"Binomial (steps={n_steps})", **gnum})

if "Finite Difference (CN)" in selected_methods:
    price_fn = build_price_func_for_method("FiniteDiff")
    try:
        gnum = finite_diff_greeks(price_fn, S, bump=1e-3, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type)
    except Exception:
        gnum = {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan, "rho": np.nan}
    greeks_rows.append({"Method": f"FiniteDiff CN (grid={n_steps})", **gnum})

if "Monte Carlo" in selected_methods:
    price_fn = build_price_func_for_method("MonteCarlo")
    try:
        gnum = finite_diff_greeks(price_fn, S, bump=1e-2, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type)
    except Exception:
        gnum = {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan, "rho": np.nan}
    greeks_rows.append({"Method": f"Monte Carlo (approx)", **gnum})

greeks_df = pd.DataFrame(greeks_rows).set_index("Method")

# format to 4 decimals and center cells
greeks_styler = greeks_df.style.format("{:.4f}") \
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
    ])

g1, g2, g3 = st.columns([1, 4, 1])
with g2:
    st.dataframe(greeks_styler, use_container_width=True)

st.markdown("---")

# ------------------------
# Greek charts + PnL plot
# ------------------------
st.header("Greeks and PnL")

# Prepare S_range and methods to plot
S_low = S * float(s_min_mult)
S_high = S * float(s_max_mult)
S_range = np.linspace(S_low, S_high, int(n_plot_points))

methods_for_plots = []
if "Black-Scholes" in selected_methods: methods_for_plots.append("Black-Scholes")
if "Binomial (CRR)" in selected_methods: methods_for_plots.append("Binomial")
if "Finite Difference (CN)" in selected_methods: methods_for_plots.append("FiniteDiff")
if ("Monte Carlo" in selected_methods) and plot_mc_with_low_paths: methods_for_plots.append("MonteCarlo")

greek_names = ["delta", "gamma", "vega", "theta", "rho"]

# accumulate arrays in dict-of-dicts
accum = {gn: {} for gn in greek_names}

# compute series (progress)
progress = st.progress(0)
total = len(methods_for_plots)
count = 0
for method in methods_for_plots:
    count += 1
    progress.progress(count / max(1, total))
    if method == "Black-Scholes":
        values_by_greek = {gn: [] for gn in greek_names}
        for s_val in S_range:
            g = bsm_greeks(s_val, K, r, q, sigma, T, option_type=option_type)
            for gn in greek_names:
                values_by_greek[gn].append(g.get(gn, np.nan))
        for gn in greek_names:
            accum[gn]["Black-Scholes"] = np.array(values_by_greek[gn])
    elif method == "Binomial":
        price_fn = build_price_func_for_method("Binomial")
        values_by_greek = {gn: [] for gn in greek_names}
        for s_val in S_range:
            try:
                gnum = finite_diff_greeks(price_fn, s_val, bump=max(1e-3, s_val * 1e-4), K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type)
            except Exception:
                gnum = {g: np.nan for g in greek_names}
            for gn in greek_names:
                values_by_greek[gn].append(gnum.get(gn, np.nan))
        for gn in greek_names:
            accum[gn][f"Binomial (steps={n_steps})"] = np.array(values_by_greek[gn])
    elif method == "FiniteDiff":
        price_fn = build_price_func_for_method("FiniteDiff")
        values_by_greek = {gn: [] for gn in greek_names}
        for s_val in S_range:
            try:
                gnum = finite_diff_greeks(price_fn, s_val, bump=max(1e-3, s_val * 1e-4), K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type)
            except Exception:
                gnum = {g: np.nan for g in greek_names}
            for gn in greek_names:
                values_by_greek[gn].append(gnum.get(gn, np.nan))
        for gn in greek_names:
            accum[gn][f"FiniteDiff CN (grid={n_steps})"] = np.array(values_by_greek[gn])
    elif method == "MonteCarlo":
        price_fn = build_price_func_for_method("MonteCarlo")
        values_by_greek = {gn: [] for gn in greek_names}
        for s_val in S_range:
            try:
                gnum = finite_diff_greeks(price_fn, s_val, bump=max(1e-2, s_val * 1e-3), K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type)
            except Exception:
                gnum = {g: np.nan for g in greek_names}
            for gn in greek_names:
                values_by_greek[gn].append(gnum.get(gn, np.nan))
        for gn in greek_names:
            accum[gn][f"Monte Carlo (paths~{int(mc_greek_paths)})"] = np.array(values_by_greek[gn])

progress.empty()

# Create a 3x2 grid layout: two rows, each with 3 columns
row1_cols = st.columns(3)
row2_cols = st.columns(3)

# Choose figure size larger/wider than before for more readability
figsize = (5.0, 3.2)

# First row: delta, gamma, vega
for i, name in enumerate(greek_names[:3]):
    col = row1_cols[i]
    df = pd.DataFrame(accum[name], index=S_range) if accum[name] else pd.DataFrame()
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.text(0.5, 0.5, f"No data for {name}", ha="center", va="center")
    else:
        for c in df.columns:
            ax.plot(S_range, df[c], linewidth=1.6, label=c)
        ax.set_title(name.capitalize(), fontsize=11)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, linewidth=0.5)
        ax.legend(fontsize=7, loc='best')
    plt.tight_layout()
    with col:
        st.markdown(f"<div style='text-align:center;font-weight:600'>{name.capitalize()}</div>", unsafe_allow_html=True)
        st.pyplot(fig)

# Second row: theta, rho, PnL
# theta
name = greek_names[3]
col = row2_cols[0]
df = pd.DataFrame(accum[name], index=S_range) if accum[name] else pd.DataFrame()
fig, ax = plt.subplots(figsize=figsize)
if df.empty:
    ax.text(0.5, 0.5, f"No data for {name}", ha="center", va="center")
else:
    for c in df.columns:
        ax.plot(S_range, df[c], linewidth=1.6, label=c)
    ax.set_title(name.capitalize(), fontsize=11)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, linewidth=0.5)
    ax.legend(fontsize=7, loc='best')
plt.tight_layout()
with col:
    st.markdown(f"<div style='text-align:center;font-weight:600'>{name.capitalize()}</div>", unsafe_allow_html=True)
    st.pyplot(fig)

# rho
name = greek_names[4]
col = row2_cols[1]
df = pd.DataFrame(accum[name], index=S_range) if accum[name] else pd.DataFrame()
fig, ax = plt.subplots(figsize=figsize)
if df.empty:
    ax.text(0.5, 0.5, f"No data for {name}", ha="center", va="center")
else:
    for c in df.columns:
        ax.plot(S_range, df[c], linewidth=1.6, label=c)
    ax.set_title(name.capitalize(), fontsize=11)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, linewidth=0.5)
    ax.legend(fontsize=7, loc='best')
plt.tight_layout()
with col:
    st.markdown(f"<div style='text-align:center;font-weight:600'>{name.capitalize()}</div>", unsafe_allow_html=True)
    st.pyplot(fig)

# PnL in last slot (row2_cols[2])
col = row2_cols[2]
# PnL method selection small (keeps same key so UI stable)
method_for_pnl = st.selectbox("Method for PnL plot", selected_methods if selected_methods else methods_available, key="pnl_method_placement")
def pricer_wrapper(s, K=K, r=r, q=q, sigma=sigma, T=T, method=method_for_pnl, option_type=option_type):
    if method == "Black-Scholes":
        return bsm_price(s, K, r, q, sigma, T, option_type=option_type)
    elif method.startswith("Binomial"):
        return binomial_price(s, K, r, q, sigma, T, steps=n_steps, option_type=option_type)
    elif method.startswith("FiniteDiff"):
        return fd_price_cn(s, K, r, q, sigma, T, M=n_steps, N=max(10, n_steps//2), option_type=option_type)
    else:
        p, _ = mc_price(s, K, r, q, sigma, T, option_type=option_type, n_paths=max(2000,int(n_paths//10)), antithetic=antithetic, control_variate=control_variate, seed=seed)
        return p

S_range_payoff, pnl = pnl_from_pricing_method(pricer_wrapper, S, K, r, q, sigma, T,
                                              method_kwargs={"option_type": option_type},
                                              s_min_mult=s_min_mult, s_max_mult=s_max_mult, n=200)
fig, ax = plt.subplots(figsize=figsize)
ax.plot(S_range_payoff, pnl, linewidth=1.8)
ax.set_title(f"PnL ({method_for_pnl})", fontsize=11)
ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=9)
ax.grid(True, linewidth=0.5)
ax.axvline(K, color="k", linestyle="--", linewidth=0.8)
plt.tight_layout()
with col:
    st.markdown("<div style='text-align:center;font-weight:600'>PnL</div>", unsafe_allow_html=True)
    st.pyplot(fig)

st.markdown("---")
st.header("Notes")
st.write("""
- If plotting becomes slow, reduce `n_plot_points`, disable Monte Carlo greeks, or lower `n_steps`.
""")
