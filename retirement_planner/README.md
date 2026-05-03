# Bayesian Retirement Planner

A retirement Monte Carlo simulator using primary Bayesian inference with targeted frequentist diagnostics, following the academic literature on Bayesian portfolio modelling.

## Architecture

**Bayesian (primary)**
- 2-regime mixture model (bull/bear) via PyMC NUTS MCMC
- GARCH(1,1) volatility — parametric bootstrap posterior (empirical Bayes)
- Student-t fat-tail body distribution
- Forward simulation draws one posterior parameter set per path — true Bayesian predictive intervals

**Frequentist (diagnostics + tail correction)**
- Generalised Pareto Distribution (EVT) for left-tail correction — Bauwens et al. (2006) MS-GARCH-EVT architecture
- Jarque-Bera, Shapiro-Wilk normality tests
- Ljung-Box ARCH-effects test
- KS posterior predictive check

**Data**
- Primary: Yahoo Finance `^GSPC` daily (1950–today, ~18,900 rows), cached locally
- Fallback: stooq → FRED → embedded Shiller/Damodaran 1950–2024 annual dataset

## Quickstart

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/yourusername/retirement-planner
cd retirement-planner
uv sync

# Default scenario ($340K, age 35.5→100, $1500/mo contrib 35.5→45, $5K/mo withdrawal 59.5→100)
uv run python retirement_model.py

# With full frequentist diagnostics report
uv run python retirement_model.py --freq-diagnostics

# Custom scenario
uv run python retirement_model.py \
    --start 500000 --age 40 --end-age 95 \
    --contrib 2000 --contrib-from 40 --contrib-to 55 \
    --withdraw 7000 --withdraw-from 62 --withdraw-to 95 \
    --sims 50000 --chains 4 --draws 2000 --plot

# 4% rule (percentage withdrawal)
uv run python retirement_model.py --no-withdraw --withdraw-pct 0.333
```

## Docker

```bash
# Build
docker build -t retirement-planner .

# Run (market data cache persisted across runs)
docker run -v retirement-cache:/home/planner/.retirement_model_cache \
    retirement-planner \
    --start 340000 --age 35.5 --end-age 100 \
    --contrib 1500 --contrib-from 35.5 --contrib-to 45 \
    --withdraw 5000 --withdraw-from 59.5 --withdraw-to 100 \
    --sims 50000 --chains 4 --freq-diagnostics
```

## Key CLI Options

| Flag | Default | Description |
|---|---|---|
| `--start` | 340000 | Starting balance ($) |
| `--age` | 35.5 | Current age |
| `--end-age` | 100 | Target age |
| `--contrib` | 1500 | Monthly contribution ($/mo) |
| `--withdraw` | 5000 | Monthly withdrawal ($/mo) |
| `--withdraw-pct` | 0 | Monthly % of balance withdrawal |
| `--cape` | 32 | Shiller CAPE ratio |
| `--sims` | 50000 | Simulation paths |
| `--chains` | 2 | MCMC chains (4+ recommended) |
| `--draws` | 1000 | Posterior draws per chain |
| `--freq-diagnostics` | off | Print full diagnostic report |
| `--plot` | off | Save projection chart as PNG |
| `--no-live-data` | off | Use embedded dataset only |

## Roadmap

- [ ] Dash interactive UI
- [ ] Inflation-adjusted real returns mode
- [ ] Multi-asset portfolio (bonds, international)
- [ ] Sequence-of-returns risk overlay
- [ ] Docker Compose + Dash deployment

## License

MIT
