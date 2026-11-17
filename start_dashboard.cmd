@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "app.py" (
    echo Cannot find app.py beside this launcher.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the local Python environment...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv ".venv"
    ) else (
        python -m venv ".venv"
    )
    if errorlevel 1 (
        echo Python environment creation failed. Install Python 3 and try again.
        pause
        exit /b 1
    )
)

set "DASHBOARD_PYTHON=%CD%\.venv\Scripts\python.exe"
"%DASHBOARD_PYTHON%" -c "import streamlit, plotly, yfinance, pandas, numpy, scipy, pytz, requests" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dashboard dependencies...
    "%DASHBOARD_PYTHON%" -m pip install -r "requirements.txt"
    if errorlevel 1 (
        echo Dependency installation failed. Check the messages above.
        pause
        exit /b 1
    )
)

echo Starting the local dashboard. Your browser will open automatically.
echo Keep this window open while using the dashboard. Press Ctrl+C to stop it.
"%DASHBOARD_PYTHON%" -m streamlit run "app.py" --server.address 127.0.0.1 --server.headless false

if errorlevel 1 (
    echo Dashboard stopped with an error. Check the messages above.
    pause
)
