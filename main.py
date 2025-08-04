import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time
import threading
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class MarketDataManager:
    """Handles real-world market data fetching and processing"""
    
    def __init__(self, use_realtime=False, api_key=None):
        self.data_cache = {}
        self.realtime_data = {}
        self.use_realtime = use_realtime
        self.api_key = api_key  # For premium APIs like Alpha Vantage, IEX Cloud
        self.is_streaming = False
        self.stream_thread = None
        
    def fetch_stock_data(self, symbol, period="2y"):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Add technical indicators
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            self.data_cache[symbol] = data
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_realtime_price(self, symbol):
        """Get current real-time price (multiple methods)"""
        if not self.use_realtime:
            return self.get_latest_historical_price(symbol)
        
        # Method 1: Yahoo Finance (Free but limited)
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price:
                self.realtime_data[symbol] = {
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'volume': info.get('regularMarketVolume', 0)
                }
                return current_price
        except Exception as e:
            print(f"Yahoo Finance real-time error for {symbol}: {e}")
        
        # Method 2: Alpha Vantage API (requires API key)
        if self.api_key:
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'Global Quote' in data:
                    price = float(data['Global Quote']['05. price'])
                    self.realtime_data[symbol] = {
                        'price': price,
                        'timestamp': datetime.now(),
                        'volume': int(data['Global Quote']['06. volume'])
                    }
                    return price
            except Exception as e:
                print(f"Alpha Vantage real-time error for {symbol}: {e}")
        
        # Fallback to latest historical price
        return self.get_latest_historical_price(symbol)
    
    def get_latest_historical_price(self, symbol):
        """Fallback: get latest historical price"""
        if symbol in self.data_cache:
            return self.data_cache[symbol]['Close'].iloc[-1]
        return None
    
    def start_realtime_stream(self, symbols, update_interval=60):
        """Start real-time data streaming in background thread"""
        if self.is_streaming:
            print("Real-time stream already running")
            return
        
        self.is_streaming = True
        self.stream_symbols = symbols
        self.update_interval = update_interval
        
        def stream_worker():
            print(f"Starting real-time stream for {symbols}")
            while self.is_streaming:
                for symbol in self.stream_symbols:
                    try:
                        price = self.get_realtime_price(symbol)
                        if price:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: ${price:.2f}")
                    except Exception as e:
                        print(f"Stream error for {symbol}: {e}")
                
                time.sleep(self.update_interval)
        
        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()
        print(f"Real-time streaming started (updates every {update_interval}s)")
    
    def stop_realtime_stream(self):
        """Stop real-time data streaming"""
        if self.is_streaming:
            self.is_streaming = False
            print("Real-time streaming stopped")
    
    def get_market_hours_status(self):
        """Check if market is currently open (US markets)"""
        now = datetime.now()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Check if it's a weekend
        if weekday >= 5:  # Saturday or Sunday
            return False, "Market closed (Weekend)"
        
        # Check market hours (9:30 AM - 4:00 PM EST)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if market_open <= now <= market_close:
            return True, "Market open"
        elif now < market_open:
            return False, "Pre-market"
        else:
            return False, "After-hours"
    
    def get_realtime_quote(self, symbol):
        """Get comprehensive real-time quote"""
        base_price = self.get_realtime_price(symbol)
        if not base_price:
            return None
        
        # Get additional real-time data if available
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            quote = {
                'symbol': symbol,
                'current_price': base_price,
                'timestamp': datetime.now(),
                'open': info.get('regularMarketOpen'),
                'high': info.get('regularMarketDayHigh'),
                'low': info.get('regularMarketDayLow'),
                'volume': info.get('regularMarketVolume'),
                'prev_close': info.get('regularMarketPreviousClose'),
                'change': None,
                'change_percent': None
            }
            
            # Calculate change and change percent
            if quote['prev_close']:
                quote['change'] = base_price - quote['prev_close']
                quote['change_percent'] = (quote['change'] / quote['prev_close']) * 100
            
            return quote
            
        except Exception as e:
            print(f"Error getting comprehensive quote for {symbol}: {e}")
            return {
                'symbol': symbol,
                'current_price': base_price,
                'timestamp': datetime.now()
            }
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class ProbabilisticAI:
    """AI trader with probabilistic decision making"""
    
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.portfolio = {}
        self.trade_history = []
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def prepare_features(self, data, lookback=30):
        """Prepare features for ML model"""
        features = []
        targets = []
        
        for i in range(lookback, len(data) - 1):
            # Feature window
            window = data.iloc[i-lookback:i]
            
            feature_vector = [
                window['Close'].iloc[-1],
                window['Volume'].iloc[-1],
                window['MA_20'].iloc[-1],
                window['MA_50'].iloc[-1],
                window['RSI'].iloc[-1],
                window['Volatility'].iloc[-1],
                np.mean(window['Returns']),
                np.std(window['Returns'])
            ]
            
            # Remove NaN values
            if not any(pd.isna(feature_vector)):
                features.append(feature_vector)
                # Target: next day return
                next_return = data['Returns'].iloc[i+1]
                targets.append(next_return)
        
        return np.array(features), np.array(targets)
    
    def train_model(self, symbol, data):
        """Train the probabilistic model"""
        features, targets = self.prepare_features(data)
        
        if len(features) > 50:  # Minimum data requirement
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, targets)
            self.is_trained = True
            print(f"AI model trained on {len(features)} samples for {symbol}")
        else:
            print(f"Insufficient data to train model for {symbol}")
    
    def predict_return_distribution(self, current_features):
        """Predict return distribution with uncertainty"""
        if not self.is_trained:
            return 0, 0  # No prediction if not trained
        
        # Scale features
        features_scaled = self.scaler.transform([current_features])
        
        # Get prediction from each tree in the forest
        tree_predictions = []
        for tree in self.model.estimators_:
            pred = tree.predict(features_scaled)[0]
            tree_predictions.append(pred)
        
        # Calculate mean and std of predictions (uncertainty)
        mean_return = np.mean(tree_predictions)
        uncertainty = np.std(tree_predictions)
        
        return mean_return, uncertainty
    
    def make_trading_decision(self, symbol, current_data, current_price):
        """Make probabilistic trading decision"""
        if len(current_data) < 50:
            return "HOLD", 0  # Not enough data
        
        # Prepare current features
        recent_data = current_data.tail(30)
        current_features = [
            recent_data['Close'].iloc[-1],
            recent_data['Volume'].iloc[-1],
            recent_data['MA_20'].iloc[-1],
            recent_data['MA_50'].iloc[-1],
            recent_data['RSI'].iloc[-1],
            recent_data['Volatility'].iloc[-1],
            np.mean(recent_data['Returns']),
            np.std(recent_data['Returns'])
        ]
        
        # Check for NaN values
        if any(pd.isna(current_features)):
            return "HOLD", 0
        
        # Get prediction
        expected_return, uncertainty = self.predict_return_distribution(current_features)
        
        # Risk-adjusted position sizing using Kelly Criterion
        if uncertainty > 0:
            kelly_fraction = expected_return / (uncertainty ** 2)
            kelly_fraction = max(-0.25, min(0.25, kelly_fraction))  # Limit position size
        else:
            kelly_fraction = 0
        
        # Trading logic
        if expected_return > 0.01 and kelly_fraction > 0.05:  # Buy signal
            position_size = int(self.capital * kelly_fraction / current_price)
            return "BUY", position_size
        elif expected_return < -0.01 and kelly_fraction < -0.05:  # Sell signal
            current_shares = self.portfolio.get(symbol, 0)
            if current_shares > 0:
                return "SELL", current_shares
        
        return "HOLD", 0
    
    def execute_trade(self, symbol, action, quantity, price):
        """Execute a trade"""
        if action == "BUY" and quantity > 0:
            cost = quantity * price
            if cost <= self.capital:
                self.capital -= cost
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'value': cost
                })
                print(f"AI BUY: {quantity} shares of {symbol} at ${price:.2f}")
        
        elif action == "SELL" and quantity > 0:
            if self.portfolio.get(symbol, 0) >= quantity:
                revenue = quantity * price
                self.capital += revenue
                self.portfolio[symbol] -= quantity
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'value': revenue
                })
                print(f"AI SELL: {quantity} shares of {symbol} at ${price:.2f}")
    
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.capital
        for symbol, shares in self.portfolio.items():
            if symbol in current_prices:
                portfolio_value += shares * current_prices[symbol]
        return portfolio_value
    
    def get_performance_metrics(self, current_prices):
        """Calculate performance metrics"""
        total_value = self.get_portfolio_value(current_prices)
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        return {
            'total_value': total_value,
            'total_return': total_return * 100,
            'cash': self.capital,
            'portfolio': self.portfolio.copy()
        }

class TradingSimulation:
    """Main simulation environment"""
    
    def __init__(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'], use_realtime=False, api_key=None):
        self.symbols = symbols
        self.data_manager = MarketDataManager(use_realtime=use_realtime, api_key=api_key)
        self.ai_trader = ProbabilisticAI()
        self.market_data = {}
        self.use_realtime = use_realtime
        
    def initialize(self):
        """Initialize the simulation"""
        print("Initializing trading simulation...")
        
        # Check market status if using real-time
        if self.use_realtime:
            is_open, status = self.data_manager.get_market_hours_status()
            print(f"Market Status: {status}")
        
        # Fetch market data
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            data = self.data_manager.fetch_stock_data(symbol)
            if data is not None:
                self.market_data[symbol] = data
                # Train AI on historical data
                self.ai_trader.train_model(symbol, data)
        
        print(f"Simulation initialized with {len(self.market_data)} stocks")
        
        # Start real-time streaming if enabled
        if self.use_realtime:
            self.data_manager.start_realtime_stream(self.symbols, update_interval=30)
    
    def run_live_trading(self, duration_minutes=60):
        """Run live trading simulation with real-time data"""
        if not self.use_realtime:
            print("Live trading requires real-time mode. Set use_realtime=True")
            return []
        
        print(f"Starting live trading simulation for {duration_minutes} minutes")
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        performance_history = []
        
        while datetime.now() < end_time:
            current_prices = {}
            
            # Get real-time prices
            for symbol in self.symbols:
                quote = self.data_manager.get_realtime_quote(symbol)
                if quote:
                    current_prices[symbol] = quote['current_price']
                    print(f"[LIVE] {symbol}: ${quote['current_price']:.2f}")
                    
                    # AI makes trading decision based on historical + current data
                    if symbol in self.market_data:
                        action, quantity = self.ai_trader.make_trading_decision(
                            symbol, self.market_data[symbol], quote['current_price']
                        )
                        
                        # Execute trade
                        if action != "HOLD":
                            self.ai_trader.execute_trade(symbol, action, quantity, quote['current_price'])
            
            # Record performance
            if current_prices:
                performance = self.ai_trader.get_performance_metrics(current_prices)
                performance['timestamp'] = datetime.now()
                performance_history.append(performance)
                
                print(f"Portfolio Value: ${performance['total_value']:,.2f} ({performance['total_return']:+.2f}%)")
            
            # Wait before next iteration
            time.sleep(60)  # Check every minute
        
        # Stop streaming
        self.data_manager.stop_realtime_stream()
        return performance_history
        
    def initialize(self):
        """Initialize the simulation"""
        print("Initializing trading simulation...")
        
        # Fetch market data
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            data = self.data_manager.fetch_stock_data(symbol)
            if data is not None:
                self.market_data[symbol] = data
                # Train AI on historical data
                self.ai_trader.train_model(symbol, data)
        
        print(f"Simulation initialized with {len(self.market_data)} stocks")
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run historical backtest"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=180)
        if not end_date:
            end_date = datetime.now() - timedelta(days=1)
        
        print(f"Running backtest from {start_date.date()} to {end_date.date()}")
        
        performance_history = []
        
        # Simulate trading day by day
        current_date = start_date
        while current_date <= end_date:
            current_prices = {}
            
            # Get current prices for all symbols
            for symbol, data in self.market_data.items():
                # Find closest date
                available_dates = data.index
                closest_date = min(available_dates, key=lambda x: abs((x.date() - current_date.date()).days))
                
                if abs((closest_date.date() - current_date.date()).days) <= 3:  # Within 3 days
                    current_price = data.loc[closest_date, 'Close']
                    current_prices[symbol] = current_price
                    
                    # Get historical data up to current date
                    historical_data = data[data.index <= closest_date]
                    
                    # AI makes trading decision
                    action, quantity = self.ai_trader.make_trading_decision(
                        symbol, historical_data, current_price
                    )
                    
                    # Execute trade
                    if action != "HOLD":
                        self.ai_trader.execute_trade(symbol, action, quantity, current_price)
            
            # Record performance
            if current_prices:
                performance = self.ai_trader.get_performance_metrics(current_prices)
                performance['date'] = current_date
                performance_history.append(performance)
            
            current_date += timedelta(days=1)
        
        return performance_history
    
    def display_results(self, performance_history):
        """Display simulation results"""
        if not performance_history:
            print("No performance data to display")
            return
        
        final_performance = performance_history[-1]
        
        print("\n" + "="*50)
        print("AI TRADING SIMULATION RESULTS")
        print("="*50)
        print(f"Initial Capital: ${self.ai_trader.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_performance['total_value']:,.2f}")
        print(f"Total Return: {final_performance['total_return']:.2f}%")
        print(f"Cash Remaining: ${final_performance['cash']:,.2f}")
        print("\nFinal Portfolio:")
        for symbol, shares in final_performance['portfolio'].items():
            if shares > 0:
                print(f"  {symbol}: {shares} shares")
        
        print(f"\nTotal Trades Executed: {len(self.ai_trader.trade_history)}")
        
        # Plot performance
        dates = [p['date'] for p in performance_history]
        values = [p['total_value'] for p in performance_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, label='AI Portfolio Value', linewidth=2)
        plt.axhline(y=self.ai_trader.initial_capital, color='r', linestyle='--', label='Initial Capital')
        plt.title('AI Trading Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Option 1: Historical backtest (no real-time)
    print("=== HISTORICAL BACKTEST ===")
    sim = TradingSimulation(['AAPL', 'GOOGL', 'MSFT'], use_realtime=False)
    sim.initialize()
    results = sim.run_backtest()
    sim.display_results(results)
    
    print("\n" + "="*50)
    
    # Option 2: Live trading simulation (with real-time data)
    print("=== LIVE TRADING SIMULATION ===")
    # Uncomment the lines below to run live simulation
    # live_sim = TradingSimulation(['AAPL', 'MSFT'], use_realtime=True)
    # live_sim.initialize()
    # live_results = live_sim.run_live_trading(duration_minutes=30)
    # live_sim.display_results(live_results)
    
    # Option 3: With premium API (Alpha Vantage)
    # Get free API key from: https://www.alphavantage.co/support/#api-key
    # premium_sim = TradingSimulation(['AAPL'], use_realtime=True, api_key='YOUR_API_KEY')
    # premium_sim.initialize()