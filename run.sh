#!/bin/bash
# Portfolio Manager - Local Run Script

echo "🚀 Starting Portfolio Manager..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Navigate to the script directory
cd "$(dirname "$0")"

# Kill any existing Streamlit processes
pkill -f streamlit 2>/dev/null

echo "📊 Portfolio Manager is starting..."
echo "🌐 Opening in browser at http://localhost:8501"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Run streamlit
streamlit run Home.py
