@echo off
REM Streamlit UI Launcher for Fraud Detection Project

echo Starting Fraud Detection Dashboard...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Start Streamlit app
streamlit run app.py --theme.base light --theme.primaryColor "#1f77b4" --theme.secondaryBackgroundColor "#f0f2f6"

pause
