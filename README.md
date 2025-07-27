# AI Forex Trading System

A desktop-based AI-powered forex trading system that runs locally without requiring cloud services or paid subscriptions. **Now 100% free to use with no cloud costs!**

## Features

- **Local Storage**: All data stored on your desktop (no cloud costs)
- **Demo Mode**: Works without MetaTrader 5 setup for testing
- **AI Analysis**: Uses Gemini API for market analysis (optional)
- **Paper Trading**: Safe practice trading with virtual money
- **Multiple Strategies**: Built-in trading strategies with performance tracking
- **Real-time Monitoring**: Live market analysis and trade execution
- **Risk Management**: Built-in daily loss limits and position sizing

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

**Easy way (recommended):**
```bash
python3 start_trader.py
```

**Direct way:**
```bash
python3 forex_trader2.py
```

The system will create a `config.json` file on first run. You can start using it immediately in demo mode.

## Configuration

### Basic Setup (Demo Mode)
The system works out of the box in demo mode using simulated market data. No additional setup required.

### Advanced Setup (Optional)

Edit `config.json` to customize:

```json
{
  "mt5_server": "your-mt5-server",
  "mt5_login": "your-login",
  "mt5_password": "your-password",
  "gemini_api_key": "your-gemini-api-key",
  "account_balance": 200000.0,
  "forex_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
  "min_win_rate_threshold": 0.6,
  "max_position_size": 0.02
}
```

#### MetaTrader 5 Setup (Optional)
- Download and install MetaTrader 5 from your broker
- Get your login credentials from your broker
- Update the config.json with your MT5 credentials
- If not configured, system uses demo data

#### Gemini API Setup (Optional)
- Get an API key from Google AI Studio (makersuite.google.com)
- Add it to config.json as "gemini_api_key"
- If not configured, system uses basic mock analysis

## How to Use

### Starting the System

1. Open terminal/command prompt
2. Navigate to the project folder
3. Run: `python forex_trader2.py`
4. The system will start automatically

### Available Commands

Once running, type any of these commands:

- `help` - Show all available commands
- `status` - Show current trading status and performance
- `stats` - Detailed performance statistics
- `strategies` - List all trading strategies
- `trades` - Show recent trades
- `new_strategy` - Create a new trading strategy
- `stop_learning` - Pause the AI learning
- `start_learning` - Resume AI learning
- `quit` - Stop the system

### Understanding the Output

The system will log information like:
```
2024-01-15 10:30:00 - INFO - Analyzing EURUSD: Price 1.08450, Action: BUY, Confidence: 0.75
2024-01-15 10:30:05 - INFO - PAPER TRADE EXECUTED: BUY 0.05 lots of EURUSD at 1.08450
2024-01-15 10:45:00 - INFO - Status - Win Rate: 65%, Total Profit: $1,250.00, Open Trades: 3
```

## File Structure

After running, your folder will contain:

```
├── forex_trader2.py          # Main trading system
├── config.json              # Configuration file
├── requirements.txt          # Python dependencies
├── trading_data.db          # Local database
├── ai_trader.log           # System log file
└── data/                   # Local data storage
    ├── learning/           # AI learning data
    ├── performance/        # Performance stats
    ├── strategies/         # Strategy data
    └── backups/           # Backup files
```

## Safety Features

- **Paper Trading Only**: No real money at risk
- **Daily Loss Limits**: Stops trading if losses exceed 5% per day
- **Risk Management**: Position sizing based on account balance
- **Local Storage**: All data stays on your computer
- **Demo Mode**: Works without broker connections

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **MetaTrader 5 connection fails**
   - System will use demo data automatically
   - Check MT5 credentials in config.json
   - Ensure MT5 is installed and running

3. **Gemini API errors**
   - System will use mock responses automatically
   - Check API key in config.json
   - Verify internet connection

4. **Permission errors**
   - Run terminal as administrator (Windows)
   - Check folder write permissions

### System Requirements

- Python 3.8 or higher
- Windows 10/11 (for MetaTrader 5)
- 4GB RAM minimum
- 1GB free disk space
- Internet connection (for live data and AI)

## Performance Monitoring

The system tracks:
- Win rate percentage
- Total profit/loss
- Number of trades
- Strategy performance
- Daily loss limits
- Risk metrics

Check `data/performance/current_stats.json` for detailed statistics.

## Customization

### Adding New Trading Pairs

Edit `config.json`:
```json
"forex_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"]
```

### Adjusting Risk Settings

```json
"account_balance": 100000.0,
"max_position_size": 0.01,
"min_win_rate_threshold": 0.7
```

### Creating Custom Strategies

Use the `new_strategy` command while the system is running, or edit the strategy files in `data/strategies/`.

## Legal Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading forex involves substantial risk and may result in losses. Always consult with a qualified financial advisor before making investment decisions.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `ai_trader.log`
3. Ensure all dependencies are installed correctly

The system is designed to be self-contained and work offline with demo data when external services are unavailable.