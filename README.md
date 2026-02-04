# Option Pricing: Overview & Comparison 2025

Vanilla option pricing library & interactive demo (Black-Scholes-Merton, Binomial CRR, Finite Difference (Crank-Nicolson), Monte-Carlo with Antithetic & Control Variate).

## Features

- Black-Scholes closed-form pricing + analytical Greeks.
- CRR binomial tree pricing (European).
- Finite Difference (Crank-Nicolson) solver for European options.
- Monte Carlo pricing with Antithetic variates and Control variate (discounted terminal price).
- Streamlit interactive app to compare methods, visualize PnL and greeks.
- Unit tests with `pytest`.

# Option Pricing — Overview & Comparison (2025)

Vanilla option pricing library and interactive demo. This project implements multiple numerical and analytical methods for European vanilla options and provides tools to compare their accuracy and performance.

Key methods included:

- Black-Scholes-Merton (closed-form) with analytical Greeks
- Cox-Ross-Rubinstein (CRR) Binomial tree (European)
- Finite Difference (Crank–Nicolson) PDE solver (European)
- Monte Carlo simulation with antithetic variates and control variate
- Small Streamlit app to interactively compare methods and visualize P&L and Greeks

This README keeps your original content while expanding installation, usage, development, and testing sections so contributors can get started quickly.

## Table of contents

- Features
- Quickstart (install & run)
- Usage (script + Streamlit)
- Project structure
- Development (tests, linting)
- Notes, assumptions and limitations
- Contributing

## Features

- Black-Scholes closed-form pricing + analytical Greeks
- CRR binomial tree pricing (European)
- Finite Difference (Crank–Nicolson) solver for European options
- Monte Carlo pricing with Antithetic variates and Control variate (discounted terminal price)
- Streamlit interactive app to compare pricing methods and visualize results
- Unit tests with `pytest`

## How each model works (simple explanation)

Below are brief, plain-language descriptions of each pricing method included in this project, with an intuitive explanation of how they work and their main trade-offs.

- Black-Scholes-Merton (BSM)
	- What it does: gives a closed-form formula (a direct calculation) for the price of European call and put options when the underlying follows a continuous log-normal diffusion with constant volatility and interest rate.
	- Intuition: it assumes the stock moves continuously and randomly; by solving the resulting partial differential equation you get a neat formula involving the normal distribution.
	- Pros: extremely fast, exact under the model's assumptions, gives analytical Greeks.
	- Cons: relies on idealized assumptions (constant volatility, no jumps, no dividends unless explicitly modeled).

- Cox-Ross-Rubinstein (CRR) Binomial tree
	- What it does: builds a discrete-time recombining tree of possible stock prices over N steps. At each step the price either goes up or down by fixed multipliers; option values are computed backward from expiry by risk-neutral expectation.
	- Intuition: think of time split into small slices; at each slice the stock moves up or down. By pricing from the end to the start you get a fair price today.
	- Pros: simple, flexible (can handle many payoffs), converges to BSM as N increases for European options.
	- Cons: for high accuracy you may need many steps (cost grows with N), American features require early-exercise checks.

- Finite Difference (Crank–Nicolson)
	- What it does: numerically solves the Black-Scholes PDE on a grid of stock prices and times using the Crank–Nicolson scheme, which blends explicit and implicit time-stepping for stability and accuracy.
	- Intuition: discretize the continuous PDE into a system of linear equations on a grid and march backward in time from expiry to today.
	- Pros: good accuracy for European options, flexible for boundary conditions and local volatility if extended.
	- Cons: requires building and solving linear systems, care is needed for boundary truncation, grid resolution choices affect runtime and error.

- Monte Carlo (plain, antithetic, control variate)
	- What it does: simulates many random future price paths for the underlying under the risk-neutral measure, computes the discounted payoff for each path, and averages results to estimate the option price.
	- Intuition: if you can't solve the model analytically, approximate the expected payoff by sampling many possible futures and averaging.
	- Variance reduction techniques included:
		- Antithetic variates: pair each random path with its mirror (negated random draws) so their errors partially cancel.
		- Control variate: use a related quantity with known expected value (e.g., Black-Scholes price of the same option) to reduce variance of the estimator.
	- Pros: extremely flexible (handles path-dependent payoffs, exotic features), straightforward to parallelize.
	- Cons: convergence is slow (error scales with 1/sqrt(N)); to get high precision you need many simulations unless variance reduction is used.

For practical use:

- Use Black-Scholes when the model assumptions are acceptable — it's fast and gives closed-form Greeks.
- Use Binomial for simple, robust checks and when flexibility for payoff forms is needed; increase steps until prices stabilize.
- Use Finite Difference when you need a grid-based PDE solution (or plan to handle local volatility or barriers) and want good accuracy for European payoffs.
- Use Monte Carlo for path-dependent/exotic payoffs or when other methods are infeasible; apply variance reduction and increased sample sizes for precision.


## Quickstart — install & run

Prerequisites:

- Python 3.10+ is recommended (the project was developed and tested on Python 3.11)
- Git (optional but recommended)

1) Create and activate a virtual environment (Windows PowerShell example):

Create the folder structure above.

```powershell
1. Save files with exact paths (e.g. `scripts/black_scholes.py`, `streamlit_app.py`, etc.).
2. `python -m venv .venv 
3.  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
4.   `.venv\Scripts\Activate.ps1 (after opening again VSCode, just run from this line)
5. `pip install -r requirements.txt`
6. Run tests: `pytest`
7. Start the app: `streamlit run streamlit_app.py`
```


Recommended: create a virtual environment.


2) Run unit tests:

```powershell
pytest -q
```

3) Run the Streamlit app (interactive comparison):

```powershell
streamlit run streamlit_app.py
```

If you want a simple script interface, use the CLI helper:

```powershell
python run.py --help
```
## Installation

Recommended: create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt

```

## Usage examples

From Python, you can import the library and price options directly. Example (pseudo):

```python
from src.option_pricing.models.bs import black_scholes_price

price = black_scholes_price(S=100, K=100, r=0.01, sigma=0.2, T=1.0, option_type="call")
print(price)
```

Use the Streamlit app to compare methods interactively. The app purposely exposes the main inputs (S, K, r, sigma, T) and lets you switch methods, number of steps/simulations, and toggles for variance reduction.

## Project structure

option-pricing-2025/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ main.py
├─ sripts/
│  ├─ __init__.py
│  ├─ viz.py               
│  ├─ greeks.py           # helper greeks (bump or analytic)
│  ├─ utils.py            # helpers: payoff, input validation, rng
│  └─ models/
│     ├─ __init__.py
│     ├─ binomial.py         # CRR binomial tree
│     ├─ fd.py               # Finite Difference (Crank-Nicolson)
│     ├─ mc.py               # Monte Carlo (plain, antithetic, control variate)           
│     └─ black_scholes.py    # Black-Scholes-Merton closed form + greeks
│
├─ streamlit_app.py      # interactive Streamlit UI
├─ run.py                # small CLI to price & compare methods
├─ tests/
│  ├─ test_bs.py
│  ├─ test_binomial.py
│  └─ test_mc.py

## Notes, assumptions and limitations

- Implementations focus on European vanilla options (no American exercise features)
- Binomial tree uses CRR branching and is intended for European payoffs
- Finite difference solver implements Crank–Nicolson for numerical stability; boundary conditions are simple Dirichlet/Neumann approximations depending on payoff
- Monte Carlo supports basic variance reduction (antithetic, control variate) but is not optimized for large-scale parallel runs
