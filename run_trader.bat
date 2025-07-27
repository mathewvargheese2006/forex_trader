@echo off
echo ================================
echo   AI Forex Trading System
echo ================================
echo.
echo Checking if config.json exists...
if not exist config.json (
    echo Creating config.json with your credentials...
    echo {
    echo   "mt5_server": "MetaQuotes-Demo",
    echo   "mt5_login": "94946431",
    echo   "mt5_password": "6-OuSpJy",
    echo   "gemini_api_key": "AIzaSyAYLtic94eXOIrQRaS1ojw626adcvOKNcU",
    echo   "account_balance": 200000.0,
    echo   "forex_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"],
    echo   "min_win_rate_threshold": 0.6,
    echo   "max_position_size": 0.02
    echo } > config.json
    echo Config file created!
) else (
    echo Config file found!
)
echo.
echo Starting AI Forex Trading System...
echo Type 'help' for commands once started
echo Press Ctrl+C to stop
echo.
python forex_trader2.py
pause