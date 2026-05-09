# European Option Pricer

An interactive Streamlit application for pricing European vanilla options using four numerical and analytical methods, with Greeks visualisation and PnL profiles.

**[Open the app](https://raphael-therenty-option-pricing-comparison-streamlit-app-mg3yxy.streamlit.app)**

---

## Table of contents

- [Features](#features)
- [Project structure](#project-structure)
- [Methods explained](#methods-explained)
  - [Black-Scholes-Merton](#1-black-scholes-merton)
  - [Binomial Tree (CRR)](#2-binomial-tree-crr)
  - [Finite Difference (Crank-Nicolson)](#3-finite-difference-crank-nicolson)
  - [Monte Carlo](#4-monte-carlo)
- [Performance comparison](#performance-comparison)
- [A note on Gamma across methods](#a-note-on-gamma-across-methods)

---

## Features

- **Four pricing methods** compared side by side: BSM, Binomial CRR, Finite Difference (Crank-Nicolson), Monte Carlo
- **Analytical Greeks** for BSM · numerical bump Greeks for all other methods
- **Greek charts** — Delta, Gamma, Vega, Theta, Rho plotted across a spot range
- **PnL profile** of a long position
- **Variance reduction** for Monte Carlo: antithetic variates and control variate, independently toggleable

---

## Project structure

```
european-option-pricer/
│
├── pricer/                       # core pricing library
│   ├── __init__.py               # flat public API
│   ├── greeks.py                 # bump_greeks() — shared numerical Greek estimator
│   ├── viz.py                    # compute_pnl() + plot_pnl()
│   └── models/
│       ├── __init__.py
│       ├── base.py               # abstract EuropeanOption base class
│       ├── bsm.py                # Black-Scholes-Merton + analytical Greeks
│       ├── binomial.py           # Cox-Ross-Rubinstein binomial tree
│       ├── finite_difference.py  # Crank-Nicolson PDE solver
│       └── monte_carlo.py        # Monte Carlo with antithetic & control variate
│
├── streamlit_app.py              # Streamlit UI
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Methods explained

All four methods price a European option on a stock following the log-normal dynamics:

$$dS_t = (r - q)\, S_t\, dt + \sigma\, S_t\, dW_t$$

where $S$ is the spot price, $r$ the risk-free rate, $q$ the continuous dividend yield, $\sigma$ the volatility, and $W_t$ a standard Brownian motion.

---

### 1. Black-Scholes-Merton

**Assumptions**
- Continuous trading, no transaction costs
- Constant volatility $\sigma$ and risk-free rate $r$
- Log-normally distributed stock returns
- No arbitrage

**Price formula**

For a call:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

For a put:

$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

where:

$$d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

and $N(\cdot)$ is the standard normal CDF.

**Analytical Greeks** — closed-form expressions for $\Delta$, $\Gamma$, $\mathcal{V}$, $\Theta$, $\rho$ (no numerical bumping needed).

---

### 2. Binomial Tree (CRR)

**Assumptions**
- Same as BSM but discretised in time
- At each step, the stock moves up by $u$ or down by $d$

**Construction**

Time is divided into $N$ steps of length $\Delta t = T/N$:

$$u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u}, \qquad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

Terminal payoffs at all $N+1$ leaf nodes are discounted backward using the risk-neutral probability $p$:

$$V_n = e^{-r\Delta t}\left[p\, V_{n+1}^{up} + (1-p)\, V_{n+1}^{down}\right]$$

**Greeks** — estimated by numerical bumping (symmetric finite difference on the price function).

**When to use** — simple, robust, converges to BSM as $N \to \infty$. Good for sanity-checking BSM. Increase steps for higher accuracy.

---

### 3. Finite Difference (Crank-Nicolson)

**Assumptions**
- Same as BSM — solves the BSM PDE numerically on a grid
- Dirichlet boundary conditions at $S=0$ and $S=S_{max}$

**The PDE**

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV = 0$$

The Crank-Nicolson scheme discretises this on an $(M+1) \times (N+1)$ grid by averaging the explicit and implicit Euler updates at each time step, giving a tridiagonal linear system solved at each step:

$$A\, \mathbf{V}^n = B\, \mathbf{V}^{n+1}$$

where $A$ and $B$ are tridiagonal matrices derived from the PDE coefficients. The scheme is second-order accurate in both $\Delta S$ and $\Delta t$, and unconditionally stable.

**Greeks** — estimated by numerical bumping via `_price_at_spot()`.

**When to use** — high accuracy for European options, natural extension to local-volatility models and barrier conditions. More expensive than BSM or binomial.

---

### 4. Monte Carlo

**Assumptions**
- Same log-normal dynamics as BSM
- Risk-neutral pricing: simulate under the $\mathbb{Q}$ measure

**Algorithm**

Simulate $N$ terminal stock prices in one vectorised step:

$$S_T^{(i)} = S \exp\!\left[\left(r - q - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\, Z^{(i)}\right], \quad Z^{(i)} \sim \mathcal{N}(0,1)$$

The price estimate is the average discounted payoff:

$$\hat{C} = e^{-rT}\, \frac{1}{N}\sum_{i=1}^{N} \max\!\left(S_T^{(i)} - K,\, 0\right)$$

**Variance reduction techniques**

| Technique | How it works |
|---|---|
| **Antithetic variates** | For each draw $Z$, also use $-Z$. The two estimates are negatively correlated, so their average has lower variance. No extra simulation cost. |
| **Control variate** | Use the discounted terminal stock price $e^{-rT}S_T$ as a control: its expectation $Se^{-qT}$ is known analytically. A regression coefficient $\beta$ is estimated and the payoff estimator is corrected: $\hat{C}_{cv} = \hat{C} - \beta\,(e^{-rT}\bar{S}_T - Se^{-qT})$. |

**Greeks** — estimated by numerical bumping. Use a larger bump size (1e-2) and more paths to stabilise the noisy estimator.

**When to use** — the most flexible method. Handles exotic payoffs and path-dependent features. Variance reduction is essential for acceptable accuracy at reasonable path counts.

---

## Performance comparison

Benchmarked on a single core, Python 3.11, `S=K=100, r=1%, σ=20%, T=0.5y` (averaged over multiple runs).

| Method | Config | Time per price | Std error (10k paths) |
|---|---|---|---|
| Black-Scholes-Merton | Analytic | **~0.14 ms** | N/A |
| Binomial (CRR) | 200 steps | ~0.9 ms | N/A |
| Finite Difference (CN) | 200×100 grid | ~6.5 ms | N/A |
| Monte Carlo | 10k paths, no variance reduction | ~0.4 ms | 0.0917 |
| Monte Carlo | 10k paths, antithetic only | ~0.3 ms | 0.0917 |
| Monte Carlo | 10k paths, control variate only | ~0.7 ms | **0.0429** |
| Monte Carlo | 10k paths, antithetic + CV | ~0.5 ms | **0.0427** |
| Monte Carlo | 50k paths, antithetic + CV | ~4.6 ms | ~0.019 |

**Key takeaways**

- **Antithetic variates** add almost no cost but do not reduce variance significantly here (the payoff and its mirror are not strongly negatively correlated for at-the-money options). The benefit is larger for deep in- or out-of-the-money options.
- **Control variate** halves the standard error (~0.092 → ~0.043) at roughly 2× the cost per run — a very favourable trade-off.
- **Both combined** give the same std error as CV alone, but antithetic reduces the raw path count needed (you get more independent draws from the same $N$).
- To match MC accuracy to BSM at a ~0.001 price precision, you need ~50k paths with both techniques, which takes about the same time as a single Finite Difference run.

---

## A note on Gamma across methods

You may notice **Gamma differing noticeably between BSM (analytic) and the numerical methods (Binomial, FD, MC)**. This is expected and has a precise explanation.

Gamma is the **second derivative** of the price with respect to the spot:

$$\Gamma = \frac{\partial^2 V}{\partial S^2}$$

Second derivatives are numerically **much harder to estimate accurately** than first derivatives:

- The symmetric finite-difference approximation used for bump Greeks is: $\Gamma \approx \frac{V(S+h) - 2V(S) + V(S-h)}{h^2}$
- The error in this estimate scales as $O(h^2 + \varepsilon/h^2)$, where $\varepsilon$ is the pricing error of the method. Even a small pricing error gets **divided by $h^2$**, amplifying noise dramatically.
- For **Monte Carlo**, the pricing noise $\varepsilon$ is random and of order $1/\sqrt{N}$, making bump Gamma very unstable unless you use extremely large path counts or specialised likelihood-ratio estimators.
- For **Binomial and FD**, the grid discretisation introduces a small but non-zero $\varepsilon$ that inflates Gamma estimates near the strike and at short maturities.

**In practice**: use BSM Gamma as the reference. If you need accurate numerical Gamma from other methods, increase grid resolution (more steps / grid points) and use a carefully chosen bump size $h$ — neither too large (truncation error) nor too small (amplified discretisation noise).
