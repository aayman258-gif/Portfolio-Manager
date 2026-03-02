# Regime-Aware Portfolio Manager — Product Overview

## The Problem

Individual investors managing concentrated portfolios (5–15 positions) face a fundamental challenge:

**Existing portfolio tools are built for either:**
- Professional institutions with 50+ positions and professional data feeds
- Passive investors tracking broad indexes
- Pure quant strategies ignoring fundamental quality
- Pure fundamental strategies ignoring market regime shifts

**What was missing:**
- No unified system combining quantitative regime detection with fundamental analysis
- No practical regime-driven rebalancing for personal portfolios
- No focused tools for concentrated, high-conviction portfolios with options exposure
- Too much data, not enough actionable insight ("What should I do today?")

---

## What Was Built

A 14-page Streamlit application delivering regime-aware portfolio management, options analytics, Monte Carlo simulation, fundamental analysis, and AI-assisted analysis — all from free public data sources.

### Core Workflow

Open the dashboard → In one session:

1. **Check the current market regime** (Low Vol / High Vol / Trending / Mean Reversion / Uncertain)
2. **Review portfolio health** — positions, P&L, cash balance, allocation
3. **Get actionable recommendations** — regime-adjusted scores, buy/sell/hold signals
4. **Run optimization** — target weights, rebalancing trades including cash
5. **Analyze options exposure** — live portfolio Greeks, hedge analysis, strategy builder
6. **Simulate outcomes** — Monte Carlo price paths with 3D probability distribution
7. **Research any stock** — full financial statements, 7-dimension scoring, peer comparison, screening
8. **Ask the AI assistant** — conversational analysis with portfolio context

---

## Delivered Features

### Portfolio Tracking (Home)
- Manual position entry, CSV upload, and sample data loading
- Real-time P&L using yfinance price data
- **Cash position tracking** — sidebar cash balance included in total value, allocation pie, and position table
- Options positions panel supporting single-leg and multi-leg strategies
- Portfolio persistence via JSON export/import
- Session state passed to downstream pages (prices, total value, positions)

### Market Regime Detection (Page 2)
- Five regimes: **Low Vol, High Vol, Trending, Mean Reversion, Uncertain**
- VIX level + realized volatility + entropy signals
- Regime timeline visualization and transition history
- Each regime maps to equity exposure limits used in optimization

### Position Scoring (Page 3)
- Unified score combining momentum, volatility, regime fit, and fundamentals
- Weights adapt based on current regime (trending → upweight momentum; high vol → upweight quality)

### Optimization & Rebalancing (Page 4)
- PyPortfolioOpt EfficientFrontier with Ledoit-Wolf shrinkage covariance
- **Four expected return methods:**
  - Mean Historical — simple annualized historical mean
  - EMA — exponentially weighted historical mean
  - CAPM — `rf + β × ERP` with betas estimated via OLS against SPY
  - Black-Litterman — CAPM equilibrium prior with Ledoit-Wolf covariance (shrinkage-adjusted CAPM)
- Optimization objectives: Max Sharpe, Min Volatility, Max Quadratic Utility
- Regime equity exposure caps (60–100% depending on regime)
- **Cash-aware trade calculation** — actual cash balance factored into current weights and CASH trade row
- Efficient frontier visualization; current vs. target allocation comparison

### Action Dashboard (Page 5)
- Position-level buy/sell/hold/reduce recommendations
- Regime-specific trade ideas with rationale

### Options Analytics — Portfolio-Integrated (Page 6)
- **Recommendations tab**: regime-aware strategy suggestions; portfolio context panel with equity delta and net options Greeks from open positions; contextual risk warnings; account size auto-populated from total portfolio value
- **Analyze Position tab**: toggle between Manual Input and Live Chain mode; live chain flow (ticker → expiry → call/put → strike → auto-populate form with live spot, IV, mid); Black-Scholes pricing, full Greeks output, P&L range, fair value vs. market price
- **Portfolio Greeks tab**: aggregates Greeks across all open options positions (singles + strategies); live IV from yfinance option chain with fallback to entry IV; per-position table with IV source indicator; aggregate net Δ/Γ/Θ/V/ρ with risk flags; three charts (delta by underlying, theta+vega, summary); CSV download

### Live Options Chain (Page 7)
- Real-time chain data via yfinance
- High-volume scanner, IV smile/skew analysis
- Probability analysis: PoP, expected value, touch probability (lognormal integrals)
- Multi-leg strategy builder with 8 templates: Iron Condor, Bull Call Spread, Bear Put Spread, Long Straddle, Long Strangle, Short Strangle, Iron Butterfly, Call Butterfly

### AI Assistant (Page 8)
- OpenRouter LLM with streaming responses and portfolio context injection
- **Offline mode** — no API key required; falls back to a rule-based response engine covering regime analysis, Greeks, optimization, position scoring, and general portfolio questions

### Portfolio Hedge Analyzer (Page 9)
- Auto-reads positions, cash, and prices from session state; computes portfolio beta via batch OLS vs SPY
- **Portfolio-level SPY hedges**: Protective Put, Put Spread, Collar, Put Ratio Spread — contracts sized by portfolio beta
- **Individual position hedges**: Protective Put, Collar, Covered Call per stock with live option chain; falls back to nearest expiry or Black-Scholes estimate
- Scenario analysis (−30% to +30% SPY) for all strategies; regime-aware urgency banner and recommendation

### Volatility Surface (Page 10)
- 3D implied volatility surface (strike × expiry × IV) using `go.Surface`
- 2D heatmap toggle, term structure chart, and skew analysis

### Options Flow (Page 11)
- Unusual options activity scanner
- Flow heatmap by ticker and expiry; volume/OI analysis

### Trade Suggestions (Page 12)
- Algorithm-scored trade ideas with regime filtering
- Strategy cards with risk/reward summaries

### Stock Screener & Fundamentals (Page 14)
- **Any ticker** — full income statement, balance sheet, and cash flow (annual + quarterly, 1Y–Max lookback)
- **7-dimension fundamental scoring** (0–100 each): Revenue Growth, Profitability, Return on Equity, Valuation, Leverage, Cash Generation, Earnings Quality
- Radar/spider chart with up to 5 peer overlays; composite score bar; scores written to `session_state['fundamental_scores']` for Position Scoring integration
- Trend charts per tab: margin trends, EPS, FCF margin, current ratio, debt vs cash, D&A, buybacks
- **Valuation vs Peers**: 4 multiple comparison charts, profitability grouped bar, P/E vs revenue growth bubble chart (sized by market cap), full peer metrics table, 5-year historical P/E
- **Screener**: 9 filter sliders (P/E, margins, ROE, D/E, current ratio, FCF, composite score) applied to any ticker list; results table with RdYlGn score gradient; CSV export
- Portfolio tickers auto-populated; comparison tickers pre-filled from portfolio

### Monte Carlo Simulation (Page 13)
- **Geometric Brownian Motion**: `S(t) = S0 · exp((μ − σ²/2)t + σ√t · Z)`
- Parameters (μ, σ) estimated from historical daily log-returns, annualized
- **Regime drift adjustment** — blends historical drift toward risk-free rate based on regime (High Vol = conservative, Trending = amplified, Uncertain = 50% blend)
- Fan chart with sample paths and percentile bands (p5/p25/p50/p75/p95)
- **3D price distribution surface** — probability density across time and price; 60 time snapshots × 80 price bins; cyan-to-white colorscale; toggle to 2D heatmap
- Key metrics: median price, probability of gain, VaR 95%, CVaR 95%
- Distribution histogram, percentile table (5th–95th), VaR/CVaR bar chart, return probability by bucket
- Multi-ticker overlay for side-by-side comparison

---

## Open Questions — Resolved

| Question | Resolution |
|---|---|
| Position entry method? | Manual entry + CSV upload (no broker API needed) |
| Fundamental scoring weights? | Regime-adaptive weights combining momentum, volatility, growth, quality, valuation; deep fundamental scores via Stock Screener (page 14) |
| Rebalancing frequency? | On-demand; triggered by user or regime change |
| Benchmark selection? | SPY as market proxy for CAPM betas; configurable lookback |
| Transaction costs? | Excluded from optimization (concentrated portfolio assumption) |
| Risk constraints? | 30% max single position; regime-based equity exposure cap; configurable |
| Options integration? | Full — single-leg and multi-leg positions tracked, Greeks aggregated |
| Cash tracking? | Full — cash balance in all metrics, allocation charts, and rebalancing trades |
| AI integration? | OpenRouter LLM with graceful offline fallback |

---

## Tech Stack (Delivered)

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Data | yfinance (free public API) |
| Optimization | PyPortfolioOpt (EfficientFrontier, Ledoit-Wolf, Black-Litterman) |
| Options Pricing | Black-Scholes (custom), scipy.stats |
| Simulation | NumPy Geometric Brownian Motion |
| Visualization | Plotly (2D + 3D interactive), Matplotlib |
| AI | OpenRouter API + rule-based offline fallback |
| Data Processing | pandas, numpy |

---

## Target User

**Primary:** Active individual investors managing concentrated portfolios (5–15 positions)
- Combine fundamental conviction with quantitative discipline
- Trade both equities and options
- Want regime-aware allocation guidance, not passive tracking
- Value actionable signal over metric overload

**Not for:**
- Passive index investors
- Day traders (position-level decisions, not intraday)
- Institutional investors (need compliance-grade systems)

---

*Built with the DRIVER Framework | Claude Code as Cognition Mate*

**Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>**
