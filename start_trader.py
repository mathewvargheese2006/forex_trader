#!/usr/bin/env python3
"""
Simple startup script for the AI Forex Trading System
This script provides an easy way to start the trading system with helpful information.
"""

import sys
import os
import subprocess

def main():
    print("🚀 AI Forex Trading System Launcher")
    print("=" * 50)
    
    # Check if config exists
    if not os.path.exists('config.json'):
        print("⚠️  Configuration file not found. The system will create one on first run.")
    else:
        print("✅ Configuration file found")
    
    # Check if database exists
    if os.path.exists('trading_data.db'):
        print("✅ Database file found")
    else:
        print("ℹ️  Database will be created on first run")
    
    print("\n📋 Quick Start Guide:")
    print("1. The system works immediately in DEMO mode (no real money)")
    print("2. Uses simulated market data if MetaTrader 5 is not configured")
    print("3. Type 'help' for commands once started")
    print("4. Type 'status' to see performance")
    print("5. Type 'quit' to stop the system")
    print("6. Press Ctrl+C to force stop")
    
    print("\n💡 Optional Setup:")
    print("- Edit config.json to add your MetaTrader 5 credentials")
    print("- Add Gemini API key for AI analysis (optional)")
    print("- Adjust account_balance and risk settings")
    
    print("\n🔒 Safety Features:")
    print("- Paper trading only (no real money at risk)")
    print("- Daily loss limits")
    print("- All data stored locally on your computer")
    
    print("\n" + "=" * 50)
    input("Press Enter to start the trading system...")
    
    try:
        # Start the main trading system
        subprocess.run([sys.executable, 'forex_trader2.py'])
    except KeyboardInterrupt:
        print("\n🛑 Startup cancelled by user")
    except Exception as e:
        print(f"\n❌ Error starting system: {e}")

if __name__ == "__main__":
    main()