import asyncio
import json
import logging
import threading
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
# Removed Google Cloud Storage import
import requests
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    # Create mock mt5 module for compatibility
    class MockMT5:
        TIMEFRAME_M15 = "M15"
        @staticmethod
        def initialize(): return False
        @staticmethod
        def login(*args, **kwargs): return False
        @staticmethod
        def copy_rates_from_pos(*args, **kwargs): return None
        @staticmethod
        def shutdown(): pass
    mt5 = MockMT5()
import sqlite3
from dataclasses import dataclass, asdict
import queue
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log MT5 availability after logger is configured
if not MT5_AVAILABLE:
    logger.warning("MetaTrader5 not available - running in demo mode only")

@dataclass
class Trade:
    timestamp: datetime
    pair: str
    action: str  # BUY/SELL
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    size: float = 0.0
    profit_loss: float = 0.0
    strategy: str = ""
    reasoning: str = ""
    status: str = "OPEN"  # OPEN/CLOSED/STOPPED

@dataclass
class Strategy:
    name: str
    description: str
    parameters: Dict[str, Any]
    win_rate: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    average_profit: float = 0.0
    last_updated: Optional[datetime] = None
    performance_score: float = 0.0

class ForexAITrader:
    def __init__(self, config_path: str = "config.json"):
        self.running = False
        self.learning_active = True
        self.config = self.load_config(config_path)
        
        # Account settings
        self.account_balance = self.config.get('account_balance', 200000.0)  # Configurable account size
        self.daily_loss_limit = 0.05 * self.account_balance  # 5% daily loss limit
        self.current_daily_loss = 0.0
        self.daily_reset_time = None
        
        # Trading state
        self.open_trades: List[Trade] = []
        self.trade_history: List[Trade] = []
        self.strategies: Dict[str, Strategy] = {}
        
        # API limits and usage tracking
        self.gemini_requests_today = {}  # Track requests per API key
        self.gemini_daily_limit = 1500  # Gemini free tier limit per key
        self.last_api_reset = datetime.now().date()
        self.current_api_key_index = 0  # Track which API key we're using
        
        # Communication queues
        self.chat_queue = queue.Queue()
        self.status_queue = queue.Queue()
        
        # Initialize components
        self.init_database()
        self.init_local_storage()
        self.init_mt5()
        self.load_strategies()
        
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "mt5_server": "your-mt5-server",
                "mt5_login": "your-login",
                "mt5_password": "your-password",
                "gemini_api_keys": ["your-gemini-api-key"],
                "account_balance": 200000.0,
                "forex_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
                "min_win_rate_threshold": 0.6,
                "max_position_size": 0.02
            }

    def init_database(self):
        """Initialize SQLite database for local storage"""
        self.conn = sqlite3.connect('trading_data.db', check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                pair TEXT,
                action TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                size REAL,
                profit_loss REAL,
                strategy TEXT,
                reasoning TEXT,
                status TEXT
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                name TEXT PRIMARY KEY,
                description TEXT,
                parameters TEXT,
                win_rate REAL,
                total_trades INTEGER,
                profitable_trades INTEGER,
                average_profit REAL,
                last_updated TEXT,
                performance_score REAL
            )
        ''')
        self.conn.commit()

    def init_local_storage(self):
        """Initialize local file storage directories"""
        try:
            # Create local storage directories
            storage_dirs = ['data', 'data/learning', 'data/performance', 'data/strategies', 'data/backups']
            for directory in storage_dirs:
                os.makedirs(directory, exist_ok=True)
            
            logger.info("Local storage directories initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local storage: {e}")

    def init_mt5(self):
        """Initialize MetaTrader 5 connection"""
        try:
            if not MT5_AVAILABLE:
                logger.warning("MetaTrader 5 not available - using demo mode only")
                return False
                
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Login to MT5 (only if credentials are provided)
            if (self.config['mt5_login'] != "your-login" and 
                self.config['mt5_password'] != "your-password" and
                self.config['mt5_server'] != "your-mt5-server"):
                
                if not mt5.login(
                    login=int(self.config['mt5_login']),
                    password=self.config['mt5_password'],
                    server=self.config['mt5_server']
                ):
                    logger.error("MT5 login failed - using demo data")
                    return False
                
                logger.info("MetaTrader 5 initialized and logged in successfully")
            else:
                logger.warning("MT5 credentials not configured - using demo mode")
            return True
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False

    def load_strategies(self):
        """Load strategies from database and local storage"""
        # Load from local database
        cursor = self.conn.execute("SELECT * FROM strategies")
        for row in cursor.fetchall():
            name, desc, params, win_rate, total, profitable, avg_profit, updated, score = row
            self.strategies[name] = Strategy(
                name=name,
                description=desc,
                parameters=json.loads(params),
                win_rate=win_rate,
                total_trades=total,
                profitable_trades=profitable,
                average_profit=avg_profit,
                last_updated=datetime.fromisoformat(updated) if updated else None,
                performance_score=score
            )
        
        # Initialize default strategies if none exist
        if not self.strategies:
            self.create_default_strategies()

    def create_default_strategies(self):
        """Create initial trading strategies"""
        default_strategies = [
            Strategy(
                name="Moving_Average_Crossover",
                description="Buy when fast MA crosses above slow MA, sell when opposite",
                parameters={"fast_period": 20, "slow_period": 50, "risk_reward": 2.0}
            ),
            Strategy(
                name="RSI_Momentum",
                description="Trade based on RSI overbought/oversold levels",
                parameters={"rsi_period": 14, "oversold": 30, "overbought": 70}
            ),
            Strategy(
                name="Breakout_Strategy",
                description="Trade breakouts from consolidation ranges",
                parameters={"lookback_period": 20, "breakout_threshold": 0.001}
            ),
            Strategy(
                name="Support_Resistance",
                description="Trade bounces from key support/resistance levels",
                parameters={"sr_strength": 3, "bounce_confirmation": 2}
            )
        ]
        
        for strategy in default_strategies:
            self.strategies[strategy.name] = strategy
            self.save_strategy(strategy)

    def get_next_available_api_key(self):
        """Get the next available API key in rotation, excluding those at limit"""
        api_keys = self.config.get('gemini_api_keys', [])
        if not api_keys or api_keys == ['your-gemini-api-key']:
            return None
        
        # Reset daily counters if new day
        if datetime.now().date() > self.last_api_reset:
            self.gemini_requests_today = {}
            self.last_api_reset = datetime.now().date()
            self.current_api_key_index = 0
        
        # Find available keys (not at limit)
        available_keys = []
        for i, key in enumerate(api_keys):
            usage = self.gemini_requests_today.get(key, 0)
            if usage < self.gemini_daily_limit:
                available_keys.append((i, key))
        
        if not available_keys:
            logger.warning("All Gemini API keys have reached daily limit")
            # Log status of all keys
            for i, key in enumerate(api_keys):
                usage = self.gemini_requests_today.get(key, 0)
                logger.info(f"  API Key #{i+1} (...{key[-4:]}): {usage}/{self.gemini_daily_limit} requests")
            return None
        
        # Always rotate to next available key in sequence
        # Find current position among available keys
        current_position = 0
        for i, (key_index, key) in enumerate(available_keys):
            if key_index == self.current_api_key_index:
                current_position = i
                break
        
        # Move to next available key (round-robin)
        next_position = (current_position + 1) % len(available_keys)
        next_available = available_keys[next_position]
        
        old_index = self.current_api_key_index
        self.current_api_key_index = next_available[0]
        
        # Log key rotation
        if len(available_keys) > 1:  # Only log if there are multiple keys
            logger.info(f"Rotating to API Key #{self.current_api_key_index + 1} (...{next_available[1][-4:]}) - Available: {len(available_keys)}/{len(api_keys)} keys")
        
        return next_available[1]

    async def call_gemini_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Gemini API with alternating key rotation and automatic exclusion"""
        api_key = self.get_next_available_api_key()
        
        if not api_key:
            logger.warning("No Gemini API keys available or all limits reached - using mock response")
            return self.generate_mock_response(prompt)
        
        for attempt in range(max_retries):
            try:
                response = await self.make_api_request(prompt, api_key)
                if response:
                    # Track usage for this specific key
                    if api_key not in self.gemini_requests_today:
                        self.gemini_requests_today[api_key] = 0
                    self.gemini_requests_today[api_key] += 1
                    
                    # Log which API key was used
                    api_keys = self.config.get('gemini_api_keys', [])
                    key_number = self.current_api_key_index + 1
                    logger.debug(f"Used Gemini API Key #{key_number} (...{api_key[-4:]}) - Usage: {self.gemini_requests_today[api_key]}/{self.gemini_daily_limit}")
                    
                    return response
                    
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None

    def generate_mock_response(self, prompt: str) -> str:
        """Generate mock response when API is not configured"""
        if "trading decision" in prompt.lower():
            return json.dumps({
                "action": "HOLD",
                "confidence": 0.5,
                "entry_price": 1.0000,
                "stop_loss": 0.9950,
                "take_profit": 1.0100,
                                 "reasoning": "Mock analysis - Gemini API not configured",
                "strategy_used": "Demo_Strategy",
                "risk_reward_ratio": 2.0
            })
        return "Mock response - Gemini API not configured"

    async def make_api_request(self, prompt: str, api_key: str) -> Optional[str]:
        """Make API request to Gemini for analysis"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1000,
                    "topP": 0.8,
                    "topK": 10
                }
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, json=data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content
                else:
                    logger.error("No content in Gemini response")
                    return None
            else:
                logger.error(f"Gemini API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Gemini API request error: {e}")
            return None

    def get_market_data(self, symbol: str, timeframe=mt5.TIMEFRAME_M15, count: int = 100) -> Optional[pd.DataFrame]:
        """Get market data from MT5 or generate demo data"""
        try:
            # Try to get real data from MT5
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is not None:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df
            else:
                logger.warning(f"Failed to get real rates for {symbol}, using demo data")
                return self.generate_demo_data(symbol, count)
            
        except Exception as e:
            logger.warning(f"Error getting market data for {symbol}: {e}, using demo data")
            return self.generate_demo_data(symbol, count)

    def generate_demo_data(self, symbol: str, count: int) -> pd.DataFrame:
        """Generate demo market data for testing"""
        try:
            # Generate realistic forex data
            base_price = 1.0000 if "USD" in symbol else 1.5000
            dates = pd.date_range(end=datetime.now(), periods=count, freq='15min')
            
            # Generate price data with random walk
            np.random.seed(hash(symbol) % 2**32)  # Deterministic based on symbol
            returns = np.random.normal(0, 0.0001, count)  # Small price movements
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # Create OHLC data
            df = pd.DataFrame({
                'time': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.0002))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.0002))) for p in prices],
                'close': prices,
                'tick_volume': np.random.randint(100, 1000, count),
                'spread': np.random.randint(1, 5, count),
                'real_volume': np.random.randint(1000, 10000, count)
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating demo data: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Moving averages
            df['MA_20'] = df['close'].rolling(window=20).mean()
            df['MA_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    async def analyze_market_with_ai(self, pair: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Use AI to analyze market conditions and suggest trades"""
        try:
            # Prepare market data summary
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            market_summary = f"""
            Forex Pair: {pair}
            Current Price: {latest['close']:.5f}
            Previous Price: {prev['close']:.5f}
            Change: {((latest['close'] - prev['close']) / prev['close'] * 100):.3f}%
            
            Technical Indicators:
            - MA20: {latest['MA_20']:.5f}
            - MA50: {latest['MA_50']:.5f}
            - RSI: {latest['RSI']:.2f}
            - MACD: {latest['MACD']:.5f}
            - MACD Signal: {latest['MACD_signal']:.5f}
            - Bollinger Upper: {latest['BB_upper']:.5f}
            - Bollinger Lower: {latest['BB_lower']:.5f}
            
            Recent price action (last 5 candles):
            {df[['time', 'open', 'high', 'low', 'close', 'RSI']].tail(5).to_string()}
            """
            
            prompt = f"""
            As an expert forex trader, analyze this market data and provide a trading decision.
            
            {market_summary}
            
            Consider:
            1. Current market trend and momentum
            2. Technical indicator signals
            3. Support/resistance levels
            4. Risk/reward potential
            
            Respond ONLY with a JSON object in this format:
            {{
                "action": "BUY" or "SELL" or "HOLD",
                "confidence": 0.0-1.0,
                "entry_price": price,
                "stop_loss": price,
                "take_profit": price,
                "reasoning": "detailed explanation",
                "strategy_used": "strategy name",
                "risk_reward_ratio": ratio
            }}
            
            DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.
            """
            
            response = await self.call_gemini_api(prompt)
            if response:
                try:
                    # Clean response and parse JSON
                    clean_response = response.replace('```json', '').replace('```', '').strip()
                    analysis = json.loads(clean_response)
                    return analysis
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse AI response as JSON: {e}")
                    logger.error(f"Raw response: {response}")
            
            return {"action": "HOLD", "confidence": 0.0, "reasoning": "AI analysis failed"}
            
        except Exception as e:
            logger.error(f"Error in AI market analysis: {e}")
            return {"action": "HOLD", "confidence": 0.0, "reasoning": f"Analysis error: {e}"}

    def calculate_position_size(self, pair: str, stop_loss_pips: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Risk 1% of account per trade
            risk_amount = self.account_balance * 0.01
            
            # Get pip value
            if "JPY" in pair:
                pip_value = 0.01  # For JPY pairs
            else:
                pip_value = 0.0001  # For other major pairs
            
            # Calculate position size
            position_size = risk_amount / (stop_loss_pips * pip_value * 100000)  # 100,000 is standard lot
            
            # Apply maximum position size limit
            max_size = self.account_balance * self.config.get('max_position_size', 0.02)
            position_size = min(position_size, max_size)
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default small size

    async def execute_paper_trade(self, analysis: Dict[str, Any], pair: str, current_price: float) -> Optional[Trade]:
        """Execute a paper trade based on AI analysis"""
        try:
            if analysis['action'] == 'HOLD' or analysis['confidence'] < 0.6:
                return None
            
            # Check daily loss limit
            if self.current_daily_loss >= self.daily_loss_limit:
                logger.warning("Daily loss limit reached, skipping trade")
                return None
            
            # Calculate position size
            entry_price = analysis.get('entry_price', current_price)
            stop_loss = analysis.get('stop_loss', entry_price * 0.995 if analysis['action'] == 'BUY' else entry_price * 1.005)
            take_profit = analysis.get('take_profit', entry_price * 1.01 if analysis['action'] == 'BUY' else entry_price * 0.99)
            
            stop_loss_pips = abs(entry_price - stop_loss) / (0.01 if "JPY" in pair else 0.0001)
            position_size = self.calculate_position_size(pair, stop_loss_pips)
            
            # Create trade
            trade = Trade(
                timestamp=datetime.now(),
                pair=pair,
                action=analysis['action'],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=position_size,
                strategy=analysis.get('strategy_used', 'AI_Analysis'),
                reasoning=analysis.get('reasoning', ''),
                status='OPEN'
            )
            
            self.open_trades.append(trade)
            self.save_trade(trade)
            
            logger.info(f"PAPER TRADE EXECUTED: {trade.action} {trade.size} lots of {trade.pair} at {trade.entry_price}")
            logger.info(f"Reasoning: {trade.reasoning}")
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return None

    def update_open_trades(self):
        """Update status of open trades"""
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            try:
                # Get current market data
                df = self.get_market_data(trade.pair, count=1)
                if df is None or df.empty:
                    continue
                
                current_price = df.iloc[-1]['close']
                
                # Check if trade should be closed
                trade_closed = False
                
                if trade.action == 'BUY':
                    if current_price >= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.status = 'CLOSED'
                        trade_closed = True
                    elif current_price <= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.status = 'STOPPED'
                        trade_closed = True
                else:  # SELL
                    if current_price <= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.status = 'CLOSED'
                        trade_closed = True
                    elif current_price >= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.status = 'STOPPED'
                        trade_closed = True
                
                if trade_closed:
                    # Calculate profit/loss
                    if trade.action == 'BUY':
                        trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.size * 100000
                    else:
                        trade.profit_loss = (trade.entry_price - trade.exit_price) * trade.size * 100000
                    
                    # Update daily loss
                    if trade.profit_loss < 0:
                        self.current_daily_loss += abs(trade.profit_loss)
                    
                    # Move to history
                    self.open_trades.remove(trade)
                    self.trade_history.append(trade)
                    self.update_trade_in_db(trade)
                    
                    # Update strategy performance
                    self.update_strategy_performance(trade)
                    
                    logger.info(f"TRADE CLOSED: {trade.pair} {trade.action} - P&L: ${trade.profit_loss:.2f}")
                    
            except Exception as e:
                logger.error(f"Error updating trade {trade.pair}: {e}")

    def update_strategy_performance(self, trade: Trade):
        """Update strategy performance metrics"""
        try:
            strategy_name = trade.strategy
            if strategy_name not in self.strategies:
                return
            
            strategy = self.strategies[strategy_name]
            strategy.total_trades += 1
            
            if trade.profit_loss > 0:
                strategy.profitable_trades += 1
            
            strategy.win_rate = strategy.profitable_trades / strategy.total_trades
            
            # Update average profit
            total_profit = strategy.average_profit * (strategy.total_trades - 1) + trade.profit_loss
            strategy.average_profit = total_profit / strategy.total_trades
            
            # Calculate performance score (win rate * average profit)
            strategy.performance_score = strategy.win_rate * max(strategy.average_profit, 0)
            strategy.last_updated = datetime.now()
            
            self.save_strategy(strategy)
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")

    def save_trade(self, trade: Trade):
        """Save trade to database"""
        try:
            self.conn.execute('''
                INSERT INTO trades VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.timestamp.isoformat(),
                trade.pair,
                trade.action,
                trade.entry_price,
                trade.exit_price,
                trade.stop_loss,
                trade.take_profit,
                trade.size,
                trade.profit_loss,
                trade.strategy,
                trade.reasoning,
                trade.status
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

    def update_trade_in_db(self, trade: Trade):
        """Update existing trade in database"""
        try:
            self.conn.execute('''
                UPDATE trades SET exit_price=?, profit_loss=?, status=?
                WHERE pair=? AND entry_price=? AND timestamp=?
            ''', (
                trade.exit_price,
                trade.profit_loss,
                trade.status,
                trade.pair,
                trade.entry_price,
                trade.timestamp.isoformat()
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating trade: {e}")

    def save_strategy(self, strategy: Strategy):
        """Save strategy to database"""
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO strategies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy.name,
                strategy.description,
                json.dumps(strategy.parameters),
                strategy.win_rate,
                strategy.total_trades,
                strategy.profitable_trades,
                strategy.average_profit,
                strategy.last_updated.isoformat() if strategy.last_updated else None,
                strategy.performance_score
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")

    def save_to_local(self, filename: str, data: Any):
        """Save data to local file storage"""
        try:
            # Ensure directory exists
            directory = os.path.dirname(f"data/{filename}")
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            with open(f"data/{filename}", 'w') as f:
                if isinstance(data, (dict, list)):
                    json.dump(data, f, indent=2, default=str)
                else:
                    f.write(str(data))
            
            logger.info(f"Saved {filename} to local storage")
            
        except Exception as e:
            logger.error(f"Error saving to local storage: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate and return performance statistics"""
        try:
            total_trades = len(self.trade_history)
            if total_trades == 0:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_profit": 0.0,
                    "average_profit": 0.0,
                    "best_strategy": "None",
                    "worst_strategy": "None",
                    "daily_loss": round(self.current_daily_loss, 2),
                    "daily_limit": round(self.daily_loss_limit, 2),
                    "open_trades": len(self.open_trades)
                }
            
            winning_trades = len([t for t in self.trade_history if t.profit_loss > 0])
            win_rate = winning_trades / total_trades
            total_profit = sum(t.profit_loss for t in self.trade_history)
            average_profit = total_profit / total_trades
            
            # Find best and worst strategies
            strategy_performance = {}
            for trade in self.trade_history:
                if trade.strategy not in strategy_performance:
                    strategy_performance[trade.strategy] = []
                strategy_performance[trade.strategy].append(trade.profit_loss)
            
            best_strategy = "None"
            worst_strategy = "None"
            best_performance = float('-inf')
            worst_performance = float('inf')
            
            for strategy, profits in strategy_performance.items():
                avg_performance = sum(profits) / len(profits)
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_strategy = strategy
                if avg_performance < worst_performance:
                    worst_performance = avg_performance
                    worst_strategy = strategy
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": round(win_rate * 100, 2),
                "total_profit": round(total_profit, 2),
                "average_profit": round(average_profit, 2),
                "best_strategy": best_strategy,
                "worst_strategy": worst_strategy,
                "daily_loss": round(self.current_daily_loss, 2),
                "daily_limit": round(self.daily_loss_limit, 2),
                "open_trades": len(self.open_trades),
                "account_balance": self.account_balance
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {"error": str(e)}

    async def continuous_learning_loop(self):
        """Main continuous learning and trading loop"""
        logger.info("Starting continuous learning loop...")
        
        while self.learning_active and self.running:
            try:
                # Reset daily loss counter if new day
                current_date = datetime.now().date()
                if self.daily_reset_time is None or current_date > self.daily_reset_time:
                    self.current_daily_loss = 0.0
                    self.daily_reset_time = current_date
                    logger.info("Daily loss counter reset")
                
                # Update open trades
                self.update_open_trades()
                
                # Analyze each forex pair
                for pair in self.config['forex_pairs']:
                    try:
                        # Check if we've hit daily loss limit
                        if self.current_daily_loss >= self.daily_loss_limit:
                            logger.warning(f"Daily loss limit reached: ${self.current_daily_loss:.2f}")
                            break
                        
                        # Get market data
                        df = self.get_market_data(pair)
                        if df is None or df.empty:
                            continue
                        
                        # Calculate technical indicators
                        df = self.calculate_technical_indicators(df)
                        
                        # Skip if not enough data for indicators
                        if len(df) < 50:
                            continue
                        
                        # AI analysis
                        analysis = await self.analyze_market_with_ai(pair, df)
                        
                                        # Log current activity with API key info
                current_price = df.iloc[-1]['close']
                api_key_info = f"[API Key #{self.current_api_key_index + 1}]" if self.config.get('gemini_api_keys') else "[Mock AI]"
                logger.info(f"{api_key_info} Analyzing {pair}: Price {current_price:.5f}, Action: {analysis.get('action', 'HOLD')}, Confidence: {analysis.get('confidence', 0):.2f}")
                        
                        # Execute paper trade if conditions are met
                        if analysis['action'] != 'HOLD':
                            trade = await self.execute_paper_trade(analysis, pair, current_price)
                            if trade:
                                # Save learning data to local storage
                                learning_data = {
                                    "timestamp": datetime.now().isoformat(),
                                    "pair": pair,
                                    "market_data": df.tail(5).to_dict('records'),
                                    "analysis": analysis,
                                    "trade": asdict(trade)
                                }
                                with open(f"data/learning/{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                                    json.dump(learning_data, f, indent=2)
                        
                        # Small delay between pairs
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {pair}: {e}")
                        continue
                
                # Performance analysis and strategy adaptation
                await self.analyze_and_adapt_strategies()
                
                # Save performance stats
                stats = self.get_performance_stats()
                with open(f"data/performance/current_stats.json", 'w') as f:
                    json.dump(stats, f, indent=2)
                
                # Log current status
                logger.info(f"Status - Win Rate: {stats.get('win_rate', 0)}%, "
                          f"Total Profit: ${stats.get('total_profit', 0):.2f}, "
                          f"Open Trades: {stats.get('open_trades', 0)}, "
                          f"Daily Loss: ${stats.get('daily_loss', 0):.2f}")
                
                # Wait before next iteration (15 minutes for forex analysis)
                await asyncio.sleep(900)  # 15 minutes
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def analyze_and_adapt_strategies(self):
        """Analyze strategy performance and create new ones if needed"""
        try:
            stats = self.get_performance_stats()
            current_win_rate = stats.get('win_rate', 0) / 100.0
            
            # If win rate is below threshold, try new strategies
            if current_win_rate < self.config.get('min_win_rate_threshold', 0.6) and stats.get('total_trades', 0) > 10:
                logger.info(f"Win rate {current_win_rate:.2%} below threshold, generating new strategies...")
                
                # Create new strategy using AI
                prompt = f"""
                Current trading performance needs improvement:
                - Win Rate: {current_win_rate:.2%}
                - Total Trades: {stats.get('total_trades', 0)}
                - Best Strategy: {stats.get('best_strategy', 'None')}
                - Worst Strategy: {stats.get('worst_strategy', 'None')}
                
                Create a new forex trading strategy to improve performance. Consider:
                1. Current market conditions
                2. Failed strategies to avoid
                3. Successful patterns to enhance
                4. Risk management improvements
                
                Respond ONLY with JSON:
                {{
                    "name": "strategy_name",
                    "description": "detailed description",
                    "parameters": {{
                        "key1": "value1",
                        "key2": "value2"
                    }},
                    "reasoning": "why this strategy should work better"
                }}
                
                DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.
                """
                
                response = await self.call_gemini_api(prompt)
                if response:
                    try:
                        clean_response = response.replace('```json', '').replace('```', '').strip()
                        new_strategy_data = json.loads(clean_response)
                        
                        new_strategy = Strategy(
                            name=new_strategy_data['name'],
                            description=new_strategy_data['description'],
                            parameters=new_strategy_data['parameters']
                        )
                        
                        self.strategies[new_strategy.name] = new_strategy
                        self.save_strategy(new_strategy)
                        
                        logger.info(f"Created new strategy: {new_strategy.name}")
                        logger.info(f"Reasoning: {new_strategy_data.get('reasoning', '')}")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse new strategy JSON: {e}")
            
            # Save strategy performance to local storage
            strategy_data = {strategy.name: asdict(strategy) for strategy in self.strategies.values()}
            with open(f"data/strategies/current_strategies.json", 'w') as f:
                json.dump(strategy_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error in strategy adaptation: {e}")

    async def chat_interface(self):
        """Interactive chat interface for communicating with the AI"""
        logger.info("Chat interface started. Type 'help' for commands.")
        
        while self.running:
            try:
                # Non-blocking input check
                if not self.chat_queue.empty():
                    user_input = self.chat_queue.get()
                    await self.process_chat_command(user_input)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in chat interface: {e}")

    async def process_chat_command(self, command: str):
        """Process user chat commands"""
        try:
            command = command.strip().lower()
            
            if command == 'help':
                help_text = """
                Available Commands:
                - status: Show current trading status
                - stats: Show performance statistics
                - strategies: List all strategies and their performance
                - stop_learning: Stop the learning process
                - start_learning: Start the learning process
                - improve <strategy_name>: Improve a specific strategy
                - new_strategy: Create a new trading strategy
                - trades: Show recent trades
                - quit: Shutdown the system
                """
                logger.info(help_text)
                
            elif command == 'status':
                stats = self.get_performance_stats()
                status_msg = f"""
                Current Status:
                - System Running: {self.running}
                - Learning Active: {self.learning_active}
                - Open Trades: {stats.get('open_trades', 0)}
                - Win Rate: {stats.get('win_rate', 0)}%
                - Total Profit: ${stats.get('total_profit', 0):.2f}
                - Daily Loss: ${stats.get('daily_loss', 0):.2f} / ${stats.get('daily_limit', 0):.2f}
                - API Calls Today: {sum(self.gemini_requests_today.values())} total"""
                
                # Show status for each API key
                api_keys = self.config.get('gemini_api_keys', [])
                available_count = 0
                for i, key in enumerate(api_keys):
                    usage = self.gemini_requests_today.get(key, 0)
                    status = "ACTIVE" if i == self.current_api_key_index else "READY"
                    if usage >= self.gemini_daily_limit:
                        status = "EXCLUDED"
                    else:
                        available_count += 1
                    status_msg += f"\n                - API Key #{i+1} (...{key[-4:]}): {usage}/{self.gemini_daily_limit} [{status}]"
                
                status_msg += f"\n                - Available Keys: {available_count}/{len(api_keys)}"
                status_msg += f"\n                - Account Balance: ${stats.get('account_balance', 0):.2f}"
                logger.info(status_msg)
                
            elif command == 'stats':
                stats = self.get_performance_stats()
                logger.info(f"Detailed Statistics: {json.dumps(stats, indent=2)}")
                
            elif command == 'strategies':
                logger.info("Current Strategies:")
                for name, strategy in self.strategies.items():
                    logger.info(f"- {name}: Win Rate {strategy.win_rate:.2%}, "
                              f"Trades: {strategy.total_trades}, "
                              f"Score: {strategy.performance_score:.2f}")
                
            elif command == 'stop_learning':
                self.learning_active = False
                logger.info("Learning process stopped")
                
            elif command == 'start_learning':
                self.learning_active = True
                logger.info("Learning process started")
                
            elif command.startswith('improve '):
                strategy_name = command.replace('improve ', '').strip()
                await self.improve_strategy(strategy_name)
                
            elif command == 'new_strategy':
                await self.create_new_strategy_interactive()
                
            elif command == 'trades':
                recent_trades = self.trade_history[-10:] if self.trade_history else []
                logger.info(f"Recent Trades ({len(recent_trades)}):")
                for trade in recent_trades:
                    logger.info(f"- {trade.pair} {trade.action} at {trade.entry_price:.5f}: "
                              f"${trade.profit_loss:.2f} ({trade.status})")
                
            elif command == 'quit':
                logger.info("Shutting down system...")
                self.running = False
                
            else:
                # Free-form chat with AI
                await self.chat_with_ai(command)
                
        except Exception as e:
            logger.error(f"Error processing command '{command}': {e}")

    async def improve_strategy(self, strategy_name: str):
        """Improve a specific strategy using AI"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"Strategy '{strategy_name}' not found")
                return
            
            strategy = self.strategies[strategy_name]
            recent_trades = [t for t in self.trade_history if t.strategy == strategy_name][-20:]
            
            prompt = f"""
            Improve the trading strategy '{strategy_name}':
            
            Current Strategy:
            - Description: {strategy.description}
            - Parameters: {json.dumps(strategy.parameters)}
            - Win Rate: {strategy.win_rate:.2%}
            - Total Trades: {strategy.total_trades}
            - Average Profit: ${strategy.average_profit:.2f}
            
            Recent Performance (last trades):
            {json.dumps([{"pair": t.pair, "action": t.action, "profit": t.profit_loss, "reasoning": t.reasoning} for t in recent_trades], indent=2)}
            
            Suggest improvements to increase win rate and profitability.
            
            Respond ONLY with JSON:
            {{
                "improved_parameters": {{"key": "value"}},
                "new_description": "updated description",
                "improvement_reasoning": "why these changes will help"
            }}
            
            DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.
            """
            
            response = await self.call_gemini_api(prompt)
            if response:
                try:
                    clean_response = response.replace('```json', '').replace('```', '').strip()
                    improvements = json.loads(clean_response)
                    
                    # Update strategy
                    strategy.parameters.update(improvements['improved_parameters'])
                    strategy.description = improvements['new_description']
                    strategy.last_updated = datetime.now()
                    
                    self.save_strategy(strategy)
                    
                    logger.info(f"Strategy '{strategy_name}' improved!")
                    logger.info(f"Reasoning: {improvements['improvement_reasoning']}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse improvement JSON: {e}")
                    
        except Exception as e:
            logger.error(f"Error improving strategy: {e}")

    async def create_new_strategy_interactive(self):
        """Create a new strategy based on current market analysis"""
        try:
            # Analyze current market conditions
            market_analysis = {}
            for pair in self.config['forex_pairs'][:3]:  # Analyze top 3 pairs
                df = self.get_market_data(pair)
                if df is not None and len(df) > 50:
                    df = self.calculate_technical_indicators(df)
                    latest = df.iloc[-1]
                    market_analysis[pair] = {
                        "price": latest['close'],
                        "trend": "UP" if latest['MA_20'] > latest['MA_50'] else "DOWN",
                        "rsi": latest['RSI'],
                        "volatility": df['close'].pct_change().std() * 100
                    }
            
            prompt = f"""
            Create a new forex trading strategy based on current market analysis:
            
            Market Conditions:
            {json.dumps(market_analysis, indent=2)}
            
            Current System Performance:
            {json.dumps(self.get_performance_stats(), indent=2)}
            
            Design a strategy that adapts to current market conditions and addresses performance gaps.
            
            Respond ONLY with JSON:
            {{
                "name": "unique_strategy_name",
                "description": "detailed strategy description",
                "parameters": {{
                    "entry_conditions": ["condition1", "condition2"],
                    "exit_conditions": ["condition1", "condition2"],
                    "risk_management": {{"stop_loss_pips": 20, "take_profit_pips": 40}},
                    "market_conditions": ["trending", "ranging", "volatile"]
                }},
                "expected_win_rate": 0.65,
                "reasoning": "why this strategy should work in current market"
            }}
            
            DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.
            """
            
            response = await self.call_gemini_api(prompt)
            if response:
                try:
                    clean_response = response.replace('```json', '').replace('```', '').strip()
                    strategy_data = json.loads(clean_response)
                    
                    new_strategy = Strategy(
                        name=strategy_data['name'],
                        description=strategy_data['description'],
                        parameters=strategy_data['parameters']
                    )
                    
                    self.strategies[new_strategy.name] = new_strategy
                    self.save_strategy(new_strategy)
                    
                    logger.info(f"New strategy created: {new_strategy.name}")
                    logger.info(f"Expected win rate: {strategy_data.get('expected_win_rate', 0):.1%}")
                    logger.info(f"Reasoning: {strategy_data.get('reasoning', '')}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse new strategy JSON: {e}")
                    
        except Exception as e:
            logger.error(f"Error creating new strategy: {e}")

    async def chat_with_ai(self, message: str):
        """Free-form chat with AI about trading"""
        try:
            context = {
                "current_stats": self.get_performance_stats(),
                "active_strategies": list(self.strategies.keys()),
                "open_trades": len(self.open_trades),
                "recent_performance": [
                    {"pair": t.pair, "profit": t.profit_loss, "strategy": t.strategy} 
                    for t in self.trade_history[-5:]
                ]
            }
            
            prompt = f"""
            You are an expert forex trading AI assistant. The user wants to discuss: "{message}"
            
            Current Trading Context:
            {json.dumps(context, indent=2)}
            
            Provide helpful, actionable advice about forex trading, strategy improvement, or system optimization.
            Be conversational and insightful.
            """
            
            response = await self.call_gemini_api(prompt)
            if response:
                logger.info(f"AI Response: {response}")
            else:
                logger.info("AI is currently unavailable for chat.")
                
        except Exception as e:
            logger.error(f"Error in AI chat: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.learning_active = False

    async def run_system(self):
        """Main system runner"""
        try:
            self.running = True
            logger.info("🚀 AI Forex Trading System Starting...")
            logger.info(f"Account Balance: ${self.account_balance:,.2f}")
            logger.info(f"Daily Loss Limit: ${self.daily_loss_limit:,.2f}")
            logger.info(f"Monitoring pairs: {', '.join(self.config['forex_pairs'])}")
        
        # Log API key status
        api_keys = self.config.get('gemini_api_keys', [])
        if api_keys and api_keys != ['your-gemini-api-key']:
            logger.info(f"Gemini API Keys configured: {len(api_keys)} keys available")
            for i, key in enumerate(api_keys):
                logger.info(f"  API Key #{i+1}: ...{key[-4:]} (Limit: {self.gemini_daily_limit}/day)")
        else:
            logger.info("No Gemini API keys configured - using mock AI responses")
            
            # Create tasks for concurrent execution
            tasks = [
                asyncio.create_task(self.continuous_learning_loop()),
                asyncio.create_task(self.chat_interface())
            ]
            
            # Start input thread for chat
            input_thread = threading.Thread(target=self.input_handler, daemon=True)
            input_thread.start()
            
            # Run all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.cleanup()

    def input_handler(self):
        """Handle user input in separate thread"""
        while self.running:
            try:
                user_input = input()
                self.chat_queue.put(user_input)
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            except Exception as e:
                logger.error(f"Input handler error: {e}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up resources...")
            
            # Close database connection
            if hasattr(self, 'conn'):
                self.conn.close()
            
            # Shutdown MT5
            if MT5_AVAILABLE and mt5.initialize():
                mt5.shutdown()
            
            # Save final state to local storage
            final_stats = self.get_performance_stats()
            with open(f"data/final_session_stats.json", 'w') as f:
                json.dump(final_stats, f, indent=2)
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Main execution
if __name__ == "__main__":
    # Create config file if it doesn't exist
    config_file = "config.json"
    if not os.path.exists(config_file):
        default_config = {
            "mt5_server": "your-mt5-server",
            "mt5_login": "your-login", 
            "mt5_password": "your-password",
            "gemini_api_keys": ["your-gemini-api-key"],
            "account_balance": 200000.0,
            "forex_pairs": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"],
            "min_win_rate_threshold": 0.6,
            "max_position_size": 0.02
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print("🔧 Created config.json. Please update it with your API credentials.")
        print("\nRequired updates:")
        print("1. MT5 server, login, and password")
        print("2. Gemini API key")
        print("3. Adjust account_balance if needed")
        sys.exit(1)
    
    # Initialize and run the trading system
    trader = ForexAITrader(config_file)
    
    try:
        print("🚀 Starting AI Forex Trading System...")
        print("Type 'help' for available commands once started.")
        print("Press Ctrl+C to stop the system.")
        
        # Run the system
        asyncio.run(trader.run_system())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 AI Forex Trading System stopped.")