# Regime-Aware Portfolio Manager

A comprehensive portfolio management system that combines regime detection, options trading analytics, and multi-leg options strategy building to maximize profitability.

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

## Features

### 1. Portfolio Position Tracker
- **Manual entry** - Enter positions directly on the website
- Load positions from CSV or sample data
- Real-time P&L tracking
- Holdings breakdown and allocation analysis
- Performance metrics and analytics

### 2. Market Regime Dashboard
- Automated regime detection (Low Vol, High Vol, Trending, Mean Reversion)
- VIX monitoring and realized volatility analysis
- Entropy-based sample space expansion detection
- Visual regime timeline and transitions

### 3. Position Scoring Engine
- Unified scoring combining quantitative and fundamental metrics
- Regime-aware weighting (adapts based on market conditions)
- Momentum, volatility, and regime fit analysis
- Growth, quality, and valuation scoring

### 4. Optimization & Rebalancing
- Regime-aware portfolio optimization using PyPortfolioOpt
- Multiple optimization methods (Max Sharpe, Min Volatility, Risk Parity)
- Dynamic equity exposure constraints based on regime
- Efficient frontier visualization

### 5. Action Dashboard
- Actionable recommendations (Buy, Sell, Hold, Reduce)
- Position-level analysis with scores and targets
- Regime-specific trade ideas

### 6. Options Analytics
- Black-Scholes options pricing
- Full Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Regime-aware strategy recommendations
- Manual options calculator

### 7. Live Options Chain
- Real-time options data via yfinance API
- Full options chain with filtering
- High volume options scanner
- Implied volatility smile/skew analysis
- Quick strategy analyzer
- **Multi-leg strategy builder** with 8 pre-built templates

## Technology Stack

- **UI Framework**: Streamlit
- **Data Sources**: yfinance (free public API)
- **Portfolio Optimization**: PyPortfolioOpt
- **Options Pricing**: Black-Scholes with custom implementation
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/portfolio-manager.git
cd portfolio-manager
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run Home.py
```

## Usage

### Loading Your Portfolio

**Option 1: Manual Entry (Recommended)**
- Enter positions directly on the website
- Add ticker, shares, and cost basis one by one
- Perfect for quick portfolio setup

**Option 2: CSV Upload**
- Upload a CSV file with columns: `ticker`, `shares`, `cost_basis`, `purchase_date`
- Great for bulk imports

**Option 3: Sample Data**
- Use the sample portfolio to explore features

### Analyzing Market Regime
Navigate to "Market Regime Dashboard" to see current market conditions and how they affect your strategy.

### Building Options Strategies
1. Go to "Live Options Chain"
2. Enter a ticker symbol
3. Navigate to "Strategy Builder" tab
4. Load a template (e.g., Iron Condor) or build custom strategies
5. Analyze payoff diagrams, Greeks, and risk/reward

### Portfolio Optimization
1. Go to "Optimization & Rebalancing"
2. Select optimization method
3. Review recommended weights and efficient frontier
4. Compare to current allocation

## Strategy Templates

The multi-leg strategy builder includes:
- **Iron Condor** - Sell OTM put spread + OTM call spread
- **Bull Call Spread** - Buy ATM call, sell OTM call
- **Bear Put Spread** - Buy ATM put, sell OTM put
- **Long Straddle** - Buy ATM call and put
- **Long Strangle** - Buy OTM call and put
- **Short Strangle** - Sell OTM call and put
- **Iron Butterfly** - Sell ATM straddle, buy OTM wings
- **Call Butterfly** - Buy lower call, sell 2x ATM calls, buy upper call

## Project Structure

```
portfolio-manager/
├── Home.py                          # Main entry point
├── pages/
│   ├── 1_Market_Regime.py
│   ├── 2_Position_Scoring.py
│   ├── 3_Optimization.py
│   ├── 4_Action_Dashboard.py
│   ├── 6_Options_Analytics.py
│   └── 7_Live_Options_Chain.py
├── calculations/
│   ├── regime_detector.py          # Regime classification logic
│   ├── scoring_engine.py           # Position scoring
│   ├── optimizer.py                # Portfolio optimization
│   ├── options_analytics.py        # Black-Scholes & Greeks
│   ├── options_recommender.py      # Strategy recommendations
│   └── strategy_builder.py         # Multi-leg strategy builder
├── data/
│   ├── portfolio_loader.py         # Load portfolio data
│   ├── market_data.py              # Fetch market data
│   └── options_data.py             # Live options chain data
└── product/
    ├── product-overview.md         # Product vision
    └── product-roadmap.md          # Development roadmap
```

## Regime Detection

The system classifies markets into four regimes:

- **Low Vol** - Stable, low volatility environment
- **High Vol** - Elevated volatility, higher risk
- **Trending** - Strong directional movement
- **Mean Reversion** - Choppy, range-bound

Position scoring and optimization adapt based on the current regime.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License

## Acknowledgments

Built using the DRIVER framework with Claude Code as a Cognition Mate.

---

**Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>**
