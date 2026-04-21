#!/bin/bash
# Streamlit UI Launcher for Fraud Detection Project

echo "Starting Fraud Detection Dashboard..."
echo ""

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start Streamlit app
streamlit run app.py --theme.base light --theme.primaryColor "#1f77b4" --theme.secondaryBackgroundColor "#f0f2f6"
