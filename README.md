# Regime-Aware Portfolio Manager

A comprehensive portfolio management system combining regime detection, portfolio optimization, options analytics, Monte Carlo simulation, and AI-powered analysis — built for active individual investors managing concentrated portfolios.

## Quick Start

**Run locally in 2 steps:**

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Run the app
./run.sh          # macOS/Linux
run.bat           # Windows
```

The app will open at **http://localhost:8501**

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

---

## Pages & Features

### Home — Portfolio Position Tracker
- Manual position entry (ticker, shares, cost basis, purchase date)
- CSV upload and sample portfolio load
- **Cash position tracking** — sidebar cash balance included in all portfolio metrics
- Real-time P&L, day's gain/loss, and total portfolio value (equities + cash)
- Allocation pie chart with cash slice
- Positions detail table with cost basis, market value, and P&L %
- Options positions panel (single-leg and multi-leg strategies)
- Portfolio persistence via JSON export/import

### Page 2 — Market Regime Dashboard
- Automated regime detection: **Low Vol, High Vol, Trending, Mean Reversion, Uncertain**
- VIX monitoring and realized volatility analysis
- Entropy-based sample space expansion detection
- Regime timeline chart and transition history
- Regime-specific playbook and allocation guidance

### Page 3 — Position Scoring Engine
- Unified scoring across quantitative and fundamental dimensions
- Regime-aware weighting (adapts based on current market conditions)
- Momentum, volatility, and regime-fit scoring
- Growth, quality, and valuation scoring
- Score breakdown visualization per position

### Page 4 — Optimization & Rebalancing
- Regime-aware portfolio optimization via PyPortfolioOpt
- **Multiple expected return methods**: Mean Historical, EMA, CAPM, Black-Litterman
- Optimization objectives: Max Sharpe, Min Volatility, Max Quadratic Utility
- Ledoit-Wolf shrinkage covariance estimation
- Dynamic equity exposure limits per regime (e.g., max 60% in High Vol)
- **Cash-aware rebalancing** — current cash balance factored into trade calculations
- Rebalancing trade table with BUY/SELL/INCREASE/DECREASE actions
- Current vs. target allocation comparison (pie charts)
- Efficient frontier visualization

### Page 5 — Action Dashboard
- Actionable position-level recommendations (Buy, Sell, Hold, Reduce)
- Regime-specific trade ideas and rationale
- Score-driven action prioritization

### Page 6 — Options Analytics (Portfolio-Integrated)
- **Tab 1 — Recommendations**: Regime-aware strategy suggestions; portfolio context panel showing equity delta and net options Greeks (delta, theta, vega); contextual risk warnings; account size auto-populated from portfolio value
- **Tab 2 — Analyze Position**: Toggle between Manual Input and **Live Chain mode** (select ticker → expiry → call/put → strike → auto-populate analysis form with live spot price, IV, and mid price)
- **Tab 3 — Portfolio Greeks**: Full Greeks analysis of all open options positions; live IV fetched from yfinance (falls back to entry IV); per-position table with net delta/gamma/theta/vega/rho; aggregate risk metrics and flags; charts (delta by underlying, theta+vega, portfolio summary); CSV export
- Black-Scholes pricing, full Greeks (Δ, Γ, Θ, V, ρ)
- Supports both single-leg and multi-leg strategy positions

### Page 7 — Live Options Chain
- Real-time options data via yfinance
- Full chain view with filtering by strike range and volume
- High-volume options scanner
- Implied volatility smile/skew analysis
- Probability analysis (PoP, EV, touch probability)
- **Multi-leg strategy builder** with 8 pre-built templates:
  - Iron Condor, Bull Call Spread, Bear Put Spread, Long Straddle
  - Long Strangle, Short Strangle, Iron Butterfly, Call Butterfly

### Page 8 — AI Assistant
- Conversational portfolio analysis powered by OpenRouter LLM
- Streaming responses with portfolio context injection
- **Offline mode** — when no API key is configured, falls back to a rule-based response engine covering regime, Greeks, optimization, and general portfolio questions
- Persistent chat history per session

### Page 9 — Portfolio Hedge Analyzer
- Protective put sizing and cost analysis
- Put spread construction
- Collar strategy analysis
- Regime-aware hedge recommendations

### Page 10 — Volatility Surface
- 3D implied volatility surface (strike × expiry × IV)
- 2D heatmap view toggle
- Term structure chart (IV vs. days to expiry)
- Skew analysis (OTM put vs. call IV spread)

### Page 11 — Options Flow
- Unusual options activity scanner
- Flow heatmap by ticker and expiry
- Volume/OI ratio analysis

### Page 12 — Trade Suggestions
- Algorithm-scored trade ideas
- Strategy cards with risk/reward summary
- Regime-filtered suggestions

### Page 13 — Monte Carlo Simulation
- **Geometric Brownian Motion** price simulation (`S(t) = S0 · exp((μ − σ²/2)t + σ√t · Z)`)
- Parameters estimated from historical daily log-returns (annualized μ and σ)
- **Regime drift adjustment** — blends historical drift toward risk-free rate based on current regime (e.g., High Vol → conservative, Trending → amplified)
- Fan chart with sample paths and percentile bands (p5/p25/p50/p75/p95)
- **3D price distribution surface** — probability density across time and price levels (60 time snapshots × 80 price bins), cyan-to-white colorscale; toggle to 2D heatmap
- Key outcome metrics: median price, probability of gain, VaR 95%, CVaR 95%
- Distribution histogram with percentile table (5th–95th)
- VaR/CVaR bar chart and return probability by bucket
- Multi-ticker overlay for comparison (median paths + IQR)
- Configurable: simulations (100–10k), horizon (1 week–2 years), lookback period

---

## Technology Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Data | yfinance (free public API) |
| Optimization | PyPortfolioOpt (EfficientFrontier, Ledoit-Wolf, Black-Litterman) |
| Options Pricing | Black-Scholes (custom implementation) |
| Simulation | NumPy GBM, scipy stats |
| Visualization | Plotly (interactive 2D/3D), Matplotlib |
| AI | OpenRouter API (with offline fallback) |
| Data Processing | pandas, numpy |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-manager.git
cd portfolio-manager
pip install -r requirements.txt
streamlit run Home.py
```

**Optional — AI Assistant:** Set `OPENROUTER_API_KEY` in your environment or `.env` file. Without a key, the assistant runs in offline rule-based mode.

---

## Usage

### Loading Your Portfolio
- **Manual Entry** — Add positions directly on the Home page (ticker, shares, cost basis)
- **CSV Upload** — Columns: `ticker`, `shares`, `cost_basis`, `purchase_date`
- **Sample Data** — Pre-loaded portfolio to explore all features

### Tracking Cash
Enter your uninvested cash balance in the Home page sidebar. It flows through to portfolio metrics, allocation charts, and the rebalancing trade recommendations.

### Regime-Driven Workflow
1. Check **Market Regime Dashboard** → identify current regime
2. Review **Position Scoring** → see which holdings fit the regime
3. Run **Optimization & Rebalancing** → get target weights and trades
4. Consult **Action Dashboard** → prioritized buy/sell/hold list
5. Analyze options exposure on **Options Analytics** → monitor portfolio Greeks

### Building Options Strategies
1. Go to **Live Options Chain** → enter ticker
2. Navigate to **Strategy Builder** tab → load a template or build custom
3. Analyze payoff diagrams, Greeks, and risk/reward
4. Log the trade on the Home page (Options Positions panel)
5. Monitor aggregated Greeks on **Options Analytics → Portfolio Greeks**

### Monte Carlo Analysis
1. Go to **Monte Carlo Simulation**
2. Select a ticker from your portfolio or enter any symbol
3. Configure simulations, horizon, and lookback
4. Toggle regime adjustment on/off to compare scenarios
5. Inspect the 3D distribution surface to visualize path uncertainty over time

---

## Project Structure

```
portfolio-manager/
├── Home.py                              # Portfolio tracker & entry point
├── pages/
│   ├── 2_Market_Regime.py               # Regime detection dashboard
│   ├── 3_Position_Scoring.py            # Scoring engine
│   ├── 4_Optimization_Rebalancing.py    # Optimizer & trade planner
│   ├── 5_Action_Dashboard.py            # Actionable recommendations
│   ├── 6_Options_Analytics.py           # Portfolio-integrated options analytics
│   ├── 7_Live_Options_Chain.py          # Live chain + strategy builder
│   ├── 8_AI_Assistant.py                # LLM assistant (offline fallback)
│   ├── 9_Portfolio_Hedge.py             # Hedge analysis
│   ├── 10_Vol_Surface.py                # 3D volatility surface
│   ├── 11_Options_Flow.py               # Unusual flow scanner
│   ├── 12_Trade_Suggestions.py          # Algo-scored trade ideas
│   └── 13_Monte_Carlo.py                # GBM price simulation
├── calculations/
│   ├── regime_detector.py               # VIX + vol regime classification
│   ├── scoring_engine.py                # Position scoring
│   ├── optimizer.py                     # PyPortfolioOpt wrapper (CAPM, BL)
│   ├── options_analytics.py             # Black-Scholes & Greeks
│   ├── options_recommender.py           # Regime-based recommendations
│   ├── strategy_builder.py              # Multi-leg strategy templates
│   └── probability_utils.py             # PoP, EV, touch probability
├── utils/
│   ├── carbon_theme.py                  # Dark UI theme & plotly layout
│   └── theme.py                         # Legacy theme utilities
├── data/
│   ├── portfolio_loader.py              # CSV / JSON portfolio loading
│   ├── market_data.py                   # Market data helpers
│   └── options_data.py                  # Live options chain fetching
└── product/
    ├── product-overview.md              # Product description
    └── product-roadmap.md               # Development roadmap
```

---

## Regime Classification

Five regimes detected using VIX levels, realized volatility, and momentum signals:

| Regime | Description | Max Equity Exposure |
|---|---|---|
| Low Vol | Calm, stable markets | 100% |
| Trending | Strong directional move | 90% |
| Mean Reversion | Choppy, range-bound | 80% |
| Uncertain | Ambiguous signals | 65% |
| High Vol | Elevated volatility | 60% |

Optimization, scoring weights, and Monte Carlo drift all adapt based on the detected regime.

---

## Contributing

Contributions welcome. Please open an issue or submit a pull request.

## License

MIT License

---

*Built with the DRIVER framework using Claude Code as a Cognition Mate.*

**Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>**
