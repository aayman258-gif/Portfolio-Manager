@echo off
REM Portfolio Manager - Local Run Script (Windows)

echo.
echo Starting Portfolio Manager...
echo.

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
)

REM Kill any existing Streamlit processes
taskkill /F /IM streamlit.exe >nul 2>&1

echo Portfolio Manager is starting...
echo Opening in browser at http://localhost:8501
echo.
echo Press CTRL+C to stop the server
echo.

REM Run streamlit
streamlit run Home.py
