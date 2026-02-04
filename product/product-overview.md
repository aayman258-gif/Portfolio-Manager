# Regime-Aware Portfolio Manager

## The Problem

Individual investors managing concentrated portfolios (5-15 positions) face a fundamental challenge:

**Existing portfolio tools are built for either:**
- Professional institutions with 50+ positions and professional data feeds
- Passive investors tracking broad indexes
- Pure quant strategies ignoring fundamental quality
- Pure fundamental strategies ignoring market regime shifts

**What's missing:**
- No unified system combining quantitative regime detection with fundamental analysis
- No practical regime-driven rebalancing for personal portfolios
- No focused tools for concentrated, high-conviction portfolios
- Too much data, not enough actionable insight ("What should I do today?")

Current tools show you 50 metrics but don't answer: "Given the current market regime and my portfolio composition, what action should I take?"

## Success Looks Like

**Primary Output:**
Open the dashboard → See:

1. **Current market regime** (Low Vol / High Vol / Trending / Mean Reversion)
2. **Portfolio health check**
   - Current positions with P&L
   - Performance vs. benchmark
   - Risk metrics (volatility, drawdown, Sharpe)
3. **Actionable recommendations**
   - "Regime shifted to High Vol → Reduce exposure by 15%, increase cash"
   - "AAPL: Strong fundamentals but momentum weakening → Consider trim"
   - "Portfolio overweight Tech (45%) → Rebalance to 30%"
4. **Optimization suggestions**
   - Optimal allocation given current regime
   - Rebalancing trades to execute
   - Expected impact on risk/return

**Validation:**
"I open this daily and immediately know: (1) What's my regime? (2) How's my portfolio? (3) What should I do?"

No scrolling through 10 pages. No analyzing 50 metrics. **Minimal interface, maximum insight.**

## Building On (Existing Foundations)

### Portfolio Optimization:
- **PyPortfolioOpt** — Mean-variance, Black-Litterman, HRP optimization
- **skfolio** — Modern scikit-learn-based portfolio tools
- **cvxpy** — Convex optimization for custom constraints

### Fundamental Data:
- **FinanceToolkit** — Financial statements, ratios, quality metrics
- **yfinance** — Free price data and basic fundamentals
- **Alpha Vantage** — Free tier for fundamental data (25 calls/day)

### Position Tracking:
- **Dash/Plotly** — Interactive dashboards
- **pandas** — Data manipulation and analysis
- **SQLite** — Local database for position tracking

### Regime Detection:
- **Concepts from regime-trading-system** — VIX, realized vol, entropy metrics
- **Academic TAA research** — Meb Faber's quantitative approach
- **Macro regime frameworks** — Recovery, expansion, slowdown, contraction

## The Unique Part (What We're Building)

**This is NOT another portfolio tracker or optimizer.**

We're building an **intelligent portfolio advisor** that:

### 1. Regime-Aware Allocation Engine
- Detect current market regime using quantitative signals
- Map regimes to optimal portfolio characteristics:
  - **Low Vol Regime** → Quality value, dividend growth, defensive sectors
  - **High Vol Regime** → Reduce equity exposure, increase cash/bonds, quality focus
  - **Trending Regime** → Momentum stocks, sector rotation, growth tilt
  - **Mean Reversion Regime** → Contrarian plays, oversold quality names

### 2. Unified Quant + Fundamental Scoring
Each position scored on:
- **Quantitative signals:** Momentum, volatility, correlation, regime fit
- **Fundamental quality:** Revenue growth, margins, ROE, FCF, debt ratios
- **Combined score:** Weighted by regime (trending → weight momentum higher, etc.)

### 3. Concentrated Portfolio Optimization
- Optimize for 5-15 positions (not 50+)
- Position-level insights, not just portfolio-level
- Deep analysis per holding:
  - Why am I holding this?
  - Does it still fit the current regime?
  - What's the fundamental story?
  - What's the quant signal saying?

### 4. Actionable Recommendations
- **Daily:** "No action needed" vs. "Consider reducing XYZ"
- **On regime shift:** "Regime changed → Here's your new optimal allocation"
- **Rebalancing:** "Portfolio drifted → Execute these 3 trades"
- **New ideas:** "ABC meets your criteria → Consider adding"

### 5. Minimal, Focused Interface
**Single-page dashboard with 4 panels:**
- Panel 1: Market regime + key macro indicators
- Panel 2: Portfolio overview (positions, P&L, risk)
- Panel 3: Action items (what to do today)
- Panel 4: Optimization suggestions (rebalancing, new ideas)

**No overwhelming tabs. No metric overload. Just signal.**

## Tech Stack

- **UI:** Streamlit (Python end-to-end, rapid iteration, clean interface)
- **Optimization:** PyPortfolioOpt, cvxpy
- **Fundamental Data:** FinanceToolkit, yfinance, Alpha Vantage
- **Position Tracking:** SQLite (local database), pandas
- **Regime Detection:** Custom (reuse from regime-trading-system)
- **Visualization:** plotly (interactive charts)
- **Analytics:** numpy, scipy, pandas

## Asset Classes Covered

**Phase 1 (MVP):**
- Equities (individual stocks)
- ETFs
- Cash

**Phase 2 (Post-MVP):**
- Options positions (integrate with regime-trading-system)
- Bonds/Fixed Income
- Alternative assets

## Open Questions (Will Resolve During Implementation)

1. **Position entry method:** Manual CSV upload vs. broker API integration?
2. **Fundamental scoring weights:** How to combine P/E, growth, quality into single score?
3. **Rebalancing frequency:** Daily suggestions vs. weekly vs. on regime change only?
4. **Benchmark selection:** SPY default? User-configurable?
5. **Transaction costs:** Include in optimization or ignore for concentrated portfolios?
6. **Risk constraints:** Max position size? Sector limits? User-configurable?

## Target User

**Primary:** Individual investors (like you) managing concentrated portfolios
- Active investors with 5-15 high-conviction positions
- Want to combine fundamental research with quantitative discipline
- Recognize that market regimes matter for allocation
- Value actionable insights over metric overload

**Not for:**
- Passive index investors (no need for active management)
- Day traders (this is for position-level decisions, not intraday)
- Institutional investors (need professional-grade systems with compliance)

---

*Built with DRIVER Framework | Combining regime detection with fundamental analysis for concentrated portfolios*
