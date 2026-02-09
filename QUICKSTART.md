# Quick Start Guide

## Running Locally

### macOS/Linux

Simply run the provided script:

```bash
./run.sh
```

Or manually:

```bash
streamlit run Home.py
```

### Windows

Double-click `run.bat` or run from command prompt:

```cmd
run.bat
```

Or manually:

```cmd
streamlit run Home.py
```

### First Time Setup

1. **Install Python** (3.8 or higher)
   - Download from https://python.org

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   ./run.sh        # macOS/Linux
   run.bat         # Windows
   ```

4. **Access in Browser**
   - Open http://localhost:8501
   - The browser should open automatically

## Adding Your Portfolio

### Option 1: Manual Entry (Recommended)
1. Select "Manual Entry" in the sidebar
2. Enter ticker, shares, and cost basis
3. Click "Add Position"
4. Repeat for all positions

### Option 2: CSV Upload
1. Create a CSV file:
   ```csv
   ticker,shares,cost_basis,purchase_date
   AAPL,100,150.00,2023-01-15
   MSFT,50,300.00,2023-02-20
   ```
2. Select "Upload CSV" in sidebar
3. Upload your file

### Option 3: Sample Data
1. Select "Use Sample Portfolio"
2. Click "Load Sample Portfolio"
3. Explore features with demo data

## Features

- **Portfolio Tracker** - Real-time P&L and holdings
- **Market Regime** - Detect volatility and trend conditions
- **Position Scoring** - Quantitative + fundamental analysis
- **Optimization** - Regime-aware portfolio rebalancing
- **Options Analytics** - Black-Scholes pricing and Greeks
- **Live Options Chain** - Real-time options data
- **Strategy Builder** - Multi-leg options strategies

## Stopping the App

Press `CTRL+C` in the terminal where Streamlit is running.

## Troubleshooting

**Port already in use:**
```bash
pkill -f streamlit    # macOS/Linux
taskkill /F /IM streamlit.exe    # Windows
```

**Dependencies missing:**
```bash
pip install -r requirements.txt
```

**Module not found:**
Make sure you're in the `portfolio-manager` directory when running.

## Next Steps

1. Navigate through the pages in the sidebar
2. Add your positions
3. Explore market regime detection
4. Try the options strategy builder
5. Optimize your portfolio allocation

---

For detailed documentation, see `README.md`
