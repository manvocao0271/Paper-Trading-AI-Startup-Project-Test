import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import threading
import time
from collections import deque
from typing import Dict, List, Optional
import logging

# Assuming your existing classes are imported
# from your_existing_code import FastDayTradingAI, OptimizedDataManager

logger = logging.getLogger(__name__)

class RealTimeIntegratedTradingSystem:
    """
    Integration of your existing trading AI with real-time WebSocket feeds
    """
    
    def __init__(self, symbols: List[str], initial_capital: float = 25000,
                 feed_provider: str = "finnhub", api_key: str = None):
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        
        # Initialize your existing trading AI
        self.trading_ai = FastDayTradingAI(initial_capital)
        
        # Real-time data manager
        self.rt_data_manager = RealTimeDataManager(feed_provider, api_key)
        
        # Enhanced data storage for technical analysis
        self.ohlcv_data = {}  # Store OHLCV bars
        self.bar_interval = 60  # 1-minute bars
        self.max_bars = 500   # Keep 500 bars for analysis
        
        # Initialize data structures
        for symbol in symbols:
            self.ohlcv_data[symbol] = deque(maxlen=self.max_bars)
        
        # Trading state
        self.is_trading = False
        self.last_analysis_time = {}
        self.analysis_interval = 30  # Analyze every 30 seconds
        
    def start_trading_session(self):
        """Start the integrated real-time trading session"""
        logger.info("Starting integrated real-time trading session...")
        
        # Start real-time data feed
        self.rt_data_manager.start(self.symbols)
        
        # Wait for initial data
        time.sleep(10)
        
        # Load initial historical data for technical analysis
        self._load_initial_historical_data()
        
        # Start trading
        self.is_trading = True
        self._run_trading_loop()
    
    def _load_initial_historical_data(self):
        """Load initial historical data for technical indicators"""
        logger.info("Loading initial historical data...")
        
        import yfinance as yf
        
        for symbol in self.symbols:
            try:
                # Get recent historical data
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period='5d', interval='1m')
                
                if not hist_data.empty:
                    # Convert to OHLCV format and store
                    for idx, row in hist_data.iterrows():
                        bar = {
                            'timestamp': idx.to_pydatetime(),
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume']
                        }
                        self.ohlcv_data[symbol].append(bar)
                
                logger.info(f"Loaded {len(self.ohlcv_data[symbol])} historical bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
    
    def _run_trading_loop(self):
        """Main trading loop that processes real-time data"""
        logger.info("Starting main trading loop...")
        
        while self.is_trading:
            try:
                current_time = datetime.now()
                
                for symbol in self.symbols:
                    # Update OHLCV bars with real-time data
                    self._update_ohlcv_bars(symbol, current_time)
                    
                    # Check if it's time to run analysis
                    last_analysis = self.last_analysis_time.get(symbol, datetime.min)
                    if (current_time - last_analysis).total_seconds() >= self.analysis_interval:
                        
                        # Run trading analysis
                        self._analyze_and_trade(symbol, current_time)
                        self.last_analysis_time[symbol] = current_time
                
                # Check existing positions for stop-loss/take-profit
                self._check_position_exits()
                
                # Print portfolio status every minute
                if int(current_time.timestamp()) % 60 == 0:
                    self._print_portfolio_status()
                
                time.sleep(1)  # Check every second
                
            except KeyboardInterrupt:
                logger.info("Stopping trading session...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
        
        self.rt_data_manager.stop()
    
    def _update_ohlcv_bars(self, symbol: str, current_time: datetime):
        """Update OHLCV bars with real-time price data"""
        current_price = self.rt_data_manager.get_current_price(symbol)
        if current_price is None:
            return
        
        # Get recent trades for volume calculation
        recent_trades = self.rt_data_manager.get_recent_trades(symbol, 1)
        current_volume = sum(trade['size'] for trade in recent_trades)
        
        # Determine which minute bar we're in
        bar_timestamp = current_time.replace(second=0, microsecond=0)
        
        # Get the last bar
        if self.ohlcv_data[symbol] and len(self.ohlcv_data[symbol]) > 0:
            last_bar = self.ohlcv_data[symbol][-1]
            
            # If we're still in the same minute, update the current bar
            if last_bar['timestamp'] == bar_timestamp:
                last_bar['high'] = max(last_bar['high'], current_price)
                last_bar['low'] = min(last_bar['low'], current_price)
                last_bar['close'] = current_price
                last_bar['volume'] += current_volume
            else:
                # Start a new bar
                new_bar = {
                    'timestamp': bar_timestamp,
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': current_volume
                }
                self.ohlcv_data[symbol].append(new_bar)
        else:
            # First bar
            new_bar = {
                'timestamp': bar_timestamp,
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': current_volume
            }
            self.ohlcv_data[symbol].append(new_bar)
    
    def _convert_to_dataframe(self, symbol: str) -> pd.DataFrame:
        """Convert OHLCV data to DataFrame for technical analysis"""
        if not self.ohlcv_data[symbol]:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(list(self.ohlcv_data[symbol]))
        df.set_index('timestamp', inplace=True)
        
        # Rename columns to match your existing code
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators (adapted from your existing code)"""
        if len(df) < 50:  # Need minimum data for indicators
            return df
        
        # Basic indicators
        df['Returns'] = df['Close'].pct_change()
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # ATR for risk management
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # VWAP from real-time data
        vwap = self.rt_data_manager.calculate_vwap(df.index[0].strftime('%Y-%m-%d %H:%M:%S').split()[0], 60)
        if vwap:
            df['VWAP'] = vwap  # Simplified - in practice you'd calculate rolling VWAP
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (from your existing code)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _analyze_and_trade(self, symbol: str, current_time: datetime):
        """Analyze symbol and make trading decisions using real-time data"""
        try:
            # Get current price from real-time feed
            current_price = self.rt_data_manager.get_current_price(symbol)
            if current_price is None:
                return
            
            # Convert OHLCV data to DataFrame
            df = self._convert_to_dataframe(symbol)
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol} analysis")
                return
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Use your existing trading AI for decision making
            signal, position_size, stop_loss, take_profit = self.trading_ai.make_trading_decision(
                symbol, df, current_price
            )
            
            # Execute trade if signal is not HOLD
            if signal != 'HOLD' and position_size > 0:
                result = self.trading_ai.execute_trade(
                    symbol, signal, position_size, current_price, stop_loss, take_profit
                )
                
                if result:
                    logger.info(f"[{current_time.strftime('%H:%M:%S')}] EXECUTED: {result}")
                    
                    # Log additional real-time context
                    recent_trades = self.rt_data_manager.get_recent_trades(symbol, 5)
                    vwap = self.rt_data_manager.calculate_vwap(symbol, 30)
                    
                    logger.info(f"  Context: Recent trades: {len(recent_trades)}, "
                              f"VWAP: ${vwap:.2f if vwap else 0:.2f}, "
                              f"RSI: {df['RSI'].iloc[-1]:.1f}")
        
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    def _check_position_exits(self):
        """Check existing positions for stop-loss/take-profit using real-time prices"""
        positions_to_close = []
        
        for symbol, position in self.trading_ai.positions.items():
            current_price = self.rt_data_manager.get_current_price(symbol)
            if current_price is None:
                continue
            
            exit_signal = self.trading_ai.check_exits(symbol, current_price)
            if exit_signal:
                positions_to_close.append((symbol, exit_signal, current_price))
        
        # Execute exits
        for symbol, exit_type, price in positions_to_close:
            position = self.trading_ai.positions[symbol]
            result = self.trading_ai.execute_trade(
                symbol, 'SELL', position['shares'], price, 0, 0
            )
            if result:
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] {exit_type}: {result}")
    
    def _print_portfolio_status(self):
        """Print current portfolio status with real-time prices"""
        current_prices = {}
        for symbol in self.symbols:
            price = self.rt_data_manager.get_current_price(symbol)
            if price:
                current_prices[symbol] = price
        
        portfolio_value = self.trading_ai.get_portfolio_value(current_prices)
        pnl = portfolio_value - self.initial_capital
        pnl_pct = (pnl / self.initial_capital) * 100
        
        logger.info(f"Portfolio: ${portfolio_value:,.2f} ({pnl_pct:+.2f}%) | "
                   f"Cash: ${self.trading_ai.capital:,.2f} | "
                   f"Positions: {len(self.trading_ai.positions)} | "
                   f"Day Trades: {self.trading_ai.day_trades_count}/3")
        
        # Show current prices
        if current_prices:
            price_str = " | ".join([f"{s}: ${p:.2f}" for s, p in current_prices.items()])
            logger.info(f"Prices: {price_str}")
    
    def get_real_time_analytics(self, symbol: str) -> Dict:
        """Get real-time analytics for a symbol"""
        current_price = self.rt_data_manager.get_current_price(symbol)
        recent_trades = self.rt_data_manager.get_recent_trades(symbol, 5)
        vwap_5min = self.rt_data_manager.calculate_vwap(symbol, 5)
        vwap_30min = self.rt_data_manager.calculate_vwap(symbol, 30)
        
        # Get latest technical indicators
        df = self._convert_to_dataframe(symbol)
        if not df.empty and len(df) > 20:
            df = self._add_technical_indicators(df)
            latest = df.iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'recent_trade_count': len(recent_trades),
                'vwap_5min': vwap_5min,
                'vwap_30min': vwap_30min,
                'rsi': latest.get('RSI'),
                'ema_9': latest.get('EMA_9'),
                'ema_21': latest.get('EMA_21'),
                'volume_ratio': latest.get('Volume_Ratio'),
                'atr': latest.get('ATR'),
                'timestamp': datetime.now()
            }
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'recent_trade_count': len(recent_trades),
            'vwap_5min': vwap_5min,
            'vwap_30min': vwap_30min,
            'timestamp': datetime.now()
        }
    
    def stop_trading(self):
        """Stop the trading session"""
        logger.info("Stopping real-time trading session...")
        self.is_trading = False
        self.rt_data_manager.stop()

class RealTimeMarketScanner:
    """
    Real-time market scanner to identify trading opportunities
    """
    
    def __init__(self, feed_provider: str = "finnhub", api_key: str = None):
        self.rt_data_manager = RealTimeDataManager(feed_provider, api_key)
        self.scan_criteria = {
            'min_volume_ratio': 2.0,  # Minimum volume spike
            'min_price_change': 0.005,  # Minimum 0.5% price change
            'max_price': 500.0,  # Maximum price per share
            'min_price': 5.0   # Minimum price per share
        }
        self.alerts = deque(maxlen=100)
    
    def start_scanning(self, symbols: List[str]):
        """Start real-time market scanning"""
        logger.info(f"Starting market scanner for {len(symbols)} symbols...")
        self.rt_data_manager.start(symbols)
        
        # Start scanning loop
        threading.Thread(target=self._scanning_loop, daemon=True).start()
    
    def _scanning_loop(self):
        """Main scanning loop"""
        while self.rt_data_manager.is_running:
            try:
                for symbol in self.rt_data_manager.symbols:
                    self._scan_symbol(symbol)
                
                time.sleep(5)  # Scan every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in scanning loop: {e}")
                time.sleep(10)
    
    def _scan_symbol(self, symbol: str):
        """Scan individual symbol for opportunities"""
        current_price = self.rt_data_manager.get_current_price(symbol)
        if not current_price:
            return
        
        # Check price criteria
        if not (self.scan_criteria['min_price'] <= current_price <= self.scan_criteria['max_price']):
            return
        
        # Get recent trades for volume analysis
        recent_trades = self.rt_data_manager.get_recent_trades(symbol, 5)
        if len(recent_trades) < 10:  # Need sufficient trade activity
            return
        
        # Calculate volume metrics
        recent_volume = sum(trade['size'] for trade in recent_trades)
        avg_trade_size = recent_volume / len(recent_trades) if recent_trades else 0
        
        # Get price history for change calculation
        price_history = self.rt_data_manager.get_price_history(symbol, 30)
        if len(price_history) < 10:
            return
        
        # Calculate price change
        old_price = price_history[0]['price'] if price_history else current_price
        price_change_pct = abs(current_price - old_price) / old_price
        
        # Check if criteria are met
        volume_unusual = recent_volume > avg_trade_size * self.scan_criteria['min_volume_ratio']
        price_moved = price_change_pct >= self.scan_criteria['min_price_change']
        
        if volume_unusual and price_moved:
            alert = {
                'symbol': symbol,
                'price': current_price,
                'price_change_pct': price_change_pct * 100,
                'volume': recent_volume,
                'avg_volume': avg_trade_size,
                'volume_ratio': recent_volume / avg_trade_size if avg_trade_size > 0 else 0,
                'timestamp': datetime.now(),
                'alert_type': 'VOLUME_BREAKOUT'
            }
            
            self.alerts.append(alert)
            logger.info(f"ALERT: {symbol} - Price: ${current_price:.2f} ({price_change_pct*100:+.1f}%) "
                       f"Volume: {recent_volume:,} ({alert['volume_ratio']:.1f}x avg)")
    
    def get_recent_alerts(self, minutes: int = 10) -> List[Dict]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]

# Example usage and integration
if __name__ == "__main__":
    # Configuration
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    API_KEY = "your_api_key_here"  # Replace with actual API key
    INITIAL_CAPITAL = 25000
    
    print("Real-Time Integrated Trading System")
    print("=" * 50)
    
    if API_KEY == "your_api_key_here":
        print("Please add your API key to run the system!")
        print()
        print("Supported providers and how to get API keys:")
        print("1. Finnhub: https://finnhub.io/register (Free tier: 60 calls/min)")
        print("2. Polygon: https://polygon.io/pricing (Free tier: 5 calls/min)")
        print("3. Alpaca: https://alpaca.markets/ (Paper trading available)")
        exit()
    
    try:
        # Option 1: Real-time trading with your existing AI
        print("Starting real-time trading system...")
        trading_system = RealTimeIntegratedTradingSystem(
            symbols=SYMBOLS,
            initial_capital=INITIAL_CAPITAL,
            feed_provider="finnhub",  # or "polygon" or "alpaca"
            api_key=API_KEY
        )
        
        # Start trading
        trading_system.start_trading_session()
        
    except KeyboardInterrupt:
        print("\nShutting down trading system...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Option 2: Market Scanner
    # scanner = RealTimeMarketScanner("finnhub", API_KEY)
    # scanner.start_scanning(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'SPY', 'QQQ'])
    # 
    # # Let it run and check alerts
    # while True:
    #     time.sleep(60)
    #     recent_alerts = scanner.get_recent_alerts(10)
    #     if recent_alerts:
    #         print(f"Found {len(recent_alerts)} alerts in last 10 minutes")