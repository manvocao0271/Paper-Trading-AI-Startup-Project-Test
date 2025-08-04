import numpy as np
import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class OptimizedDataManager:
    """Optimized data manager with caching and batch processing"""
    
    def __init__(self):
        self.data_cache = {}
        self.price_cache = {}
        self.last_update = {}
        self.cache_duration = 60  # Cache prices for 60 seconds
        
    def fetch_all_data_batch(self, symbols):
        """Fetch all symbol data in parallel for speed"""
        print("Fetching historical data in batch...")
        
        def fetch_single(symbol):
            try:
                stock = yf.Ticker(symbol)
                # Get recent data for pattern analysis
                data = stock.history(period='5d', interval='1m')
                if not data.empty:
                    data = self.add_indicators(data)
                    return symbol, data
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
            return symbol, None
        
        # Parallel fetching for speed
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_single, symbols))
        
        for symbol, data in results:
            if data is not None:
                self.data_cache[symbol] = data
                # Cache latest price
                self.price_cache[symbol] = {
                    'price': data['Close'].iloc[-1],
                    'timestamp': datetime.now()
                }
        
        print(f"Loaded data for {len([r for r in results if r[1] is not None])} symbols")
        return self.data_cache
    
    def add_indicators(self, data):
        """Add technical indicators efficiently"""
        # Basic indicators
        data['Returns'] = data['Close'].pct_change()
        data['EMA_9'] = data['Close'].ewm(span=9).mean()
        data['EMA_21'] = data['Close'].ewm(span=21).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # ATR for risk management
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['ATR'] = true_range.rolling(window=14).mean()
        
        return data
    
    def calculate_rsi(self, prices, window=14):
        """Fast RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_cached_price(self, symbol):
        """Get cached price to avoid repeated API calls"""
        if symbol in self.price_cache:
            cache_entry = self.price_cache[symbol]
            age = (datetime.now() - cache_entry['timestamp']).total_seconds()
            
            if age < self.cache_duration:
                return cache_entry['price']
        
        # If cache expired or missing, update from historical data
        if symbol in self.data_cache:
            latest_price = self.data_cache[symbol]['Close'].iloc[-1]
            self.price_cache[symbol] = {
                'price': latest_price,
                'timestamp': datetime.now()
            }
            return latest_price
        
        return None
    
    def simulate_price_movement(self, symbol, base_price):
        """Simulate realistic price movement for demo purposes"""
        if symbol not in self.data_cache:
            return base_price
        
        # Get recent volatility
        recent_data = self.data_cache[symbol].tail(50)
        volatility = recent_data['Returns'].std()
        
        # Simulate MORE ACTIVE price movement for demo
        random_change = np.random.normal(0, volatility * 0.5)  # Increased from 0.1 to 0.5
        new_price = base_price * (1 + random_change)
        
        # Add occasional larger moves (spikes)
        if np.random.random() < 0.1:  # 10% chance of larger move
            spike = np.random.normal(0, volatility * 2)
            new_price = base_price * (1 + spike)
        
        # Update cache
        self.price_cache[symbol] = {
            'price': new_price,
            'timestamp': datetime.now()
        }
        
        return new_price

class FastDayTradingAI:
    """Optimized AI for faster decision making"""
    
    def __init__(self, initial_capital=25000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.day_trades_count = 0
        self.max_day_trades = 3
        self.trade_history = []
        
    def quick_analysis(self, symbol, data, current_price):
        """Fast trading signal analysis"""
        if len(data) < 30:
            return 'HOLD', 0
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signal_strength = 0
        signals = []
        
        # Quick momentum check
        if latest['EMA_9'] > latest['EMA_21'] and current_price > latest['EMA_9']:
            signals.append(('BUY', 0.6))
        elif latest['EMA_9'] < latest['EMA_21'] and current_price < latest['EMA_9']:
            signals.append(('SELL', 0.6))
        
        # Price direction signal (more sensitive)
        if current_price > prev['Close'] * 1.001:  # 0.1% price increase
            signals.append(('BUY', 0.4))
        elif current_price < prev['Close'] * 0.999:  # 0.1% price decrease
            signals.append(('SELL', 0.4))
        
        # Volume confirmation
        if latest['Volume_Ratio'] > 1.3:
            if current_price > prev['Close']:
                signals.append(('BUY', 0.4))
            else:
                signals.append(('SELL', 0.4))
        
        # RSI extremes (more sensitive)
        if latest['RSI'] < 40:  # Was 30
            signals.append(('BUY', 0.5))
        elif latest['RSI'] > 60:  # Was 70
            signals.append(('SELL', 0.5))
        
        # Aggregate signals
        buy_strength = sum([s[1] for s in signals if s[0] == 'BUY'])
        sell_strength = sum([s[1] for s in signals if s[0] == 'SELL'])
        
        # LOWERED THRESHOLDS for more active trading
        if buy_strength > sell_strength and buy_strength > 0.4:  # Was 0.7
            return 'BUY', buy_strength
        elif sell_strength > buy_strength and sell_strength > 0.4:  # Was 0.7
            return 'SELL', sell_strength
        
        return 'HOLD', 0
    
    def calculate_position_size(self, symbol, confidence, current_price, atr):
        """Quick position sizing"""
        if atr == 0 or pd.isna(atr):
            atr = current_price * 0.01  # 1% fallback
        
        max_risk = self.capital * 0.01  # 1% max risk
        stop_distance = atr * 2
        
        if stop_distance <= 0:
            return 0
        
        shares = int(max_risk / stop_distance)
        shares = int(shares * confidence)  # Adjust for confidence
        
        # Position size limits
        max_position = int(self.capital * 0.2 / current_price)  # Max 20% of capital
        return min(shares, max_position, 100)  # Cap at 100 shares for demo
    
    def make_trading_decision(self, symbol, data, current_price):
        """Fast trading decision"""
        # Check day trade limit
        if self.day_trades_count >= self.max_day_trades:
            return 'HOLD', 0, 0, 0
        
        # Quick analysis
        signal, confidence = self.quick_analysis(symbol, data, current_price)
        
        if signal == 'HOLD' or confidence < 0.5:
            return 'HOLD', 0, 0, 0
        
        # Get ATR for risk management
        atr = data['ATR'].iloc[-1] if not pd.isna(data['ATR'].iloc[-1]) else current_price * 0.01
        
        # Position sizing
        position_size = self.calculate_position_size(symbol, confidence, current_price, atr)
        
        if position_size <= 0:
            return 'HOLD', 0, 0, 0
        
        # Calculate stops
        if signal == 'BUY':
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        else:
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        
        return signal, position_size, stop_loss, take_profit
    
    def execute_trade(self, symbol, action, quantity, price, stop_loss, take_profit):
        """Execute trade with logging"""
        if action == 'BUY' and quantity > 0:
            cost = quantity * price
            if cost <= self.capital:
                self.capital -= cost
                self.positions[symbol] = {
                    'shares': quantity,
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now()
                }
                self.day_trades_count += 1
                
                trade_record = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'timestamp': datetime.now(),
                    'type': 'ENTRY'
                }
                self.trade_history.append(trade_record)
                
                return f"BUY {quantity} {symbol} @ ${price:.2f}"
        
        elif action == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            revenue = quantity * price
            self.capital += revenue
            
            # Calculate P&L
            cost = quantity * position['entry_price']
            pnl = revenue - cost
            
            # Remove position
            del self.positions[symbol]
            
            trade_record = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'timestamp': datetime.now(),
                'type': 'EXIT'
            }
            self.trade_history.append(trade_record)
            
            return f"SELL {quantity} {symbol} @ ${price:.2f} | P&L: ${pnl:.2f}"
        
        return None
    
    def check_exits(self, symbol, current_price):
        """Check stop loss and take profit"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if current_price <= position['stop_loss']:
            return 'STOP_LOSS'
        elif current_price >= position['take_profit']:
            return 'TAKE_PROFIT'
        
        return None
    
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        total = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position['shares'] * current_prices[symbol]
        
        return total

class FastDayTradingSimulation:
    """Optimized day trading simulation"""
    
    def __init__(self, symbols=['AAPL', 'SPY', 'QQQ'], capital=25000):
        self.symbols = symbols
        self.data_manager = OptimizedDataManager()
        self.ai_trader = FastDayTradingAI(initial_capital=capital)
        self.market_data = {}
        self.simulation_speed = 5  # 5 seconds between iterations (much faster)
        
    def initialize(self):
        """Initialize simulation with batch data loading"""
        print("Initializing optimized day trading simulation...")
        
        # Batch load all data
        self.market_data = self.data_manager.fetch_all_data_batch(self.symbols)
        
        if not self.market_data:
            print("ERROR: No market data loaded!")
            return False
        
        print(f"Simulation ready with {len(self.market_data)} symbols")
        print(f"Update frequency: Every {self.simulation_speed} seconds")
        return True
    
    def run_fast_session(self, duration_minutes=30):
        """Run optimized trading session"""
        print(f"\nStarting {duration_minutes}-minute trading session...")
        print("=" * 50)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        iteration = 0
        performance_log = []
        
        while datetime.now() < end_time:
            iteration += 1
            current_time = datetime.now()
            current_prices = {}
            
            # Get current prices (cached/simulated for speed)
            for symbol in self.symbols:
                base_price = self.data_manager.get_cached_price(symbol)
                if base_price:
                    # Simulate small price movements for demo
                    current_price = self.data_manager.simulate_price_movement(symbol, base_price)
                    current_prices[symbol] = current_price
            
            # Process each symbol
            actions_taken = []
            
            for symbol in self.symbols:
                if symbol not in current_prices or symbol not in self.market_data:
                    continue
                
                current_price = current_prices[symbol]
                
                # Check existing positions for exits
                exit_signal = self.ai_trader.check_exits(symbol, current_price)
                if exit_signal:
                    if symbol in self.ai_trader.positions:
                        position = self.ai_trader.positions[symbol]
                        result = self.ai_trader.execute_trade(
                            symbol, 'SELL', position['shares'], current_price, 0, 0
                        )
                        if result:
                            actions_taken.append(f"{exit_signal}: {result}")
                
                # Make new trading decisions
                signal, quantity, stop_loss, take_profit = self.ai_trader.make_trading_decision(
                    symbol, self.market_data[symbol], current_price
                )
                
                if signal != 'HOLD' and quantity > 0:
                    result = self.ai_trader.execute_trade(
                        symbol, signal, quantity, current_price, stop_loss, take_profit
                    )
                    if result:
                        actions_taken.append(result)
            
            # Calculate portfolio value
            portfolio_value = self.ai_trader.get_portfolio_value(current_prices)
            pnl = portfolio_value - self.ai_trader.initial_capital
            pnl_pct = (pnl / self.ai_trader.initial_capital) * 100
            
            # Log performance
            performance_log.append({
                'iteration': iteration,
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'cash': self.ai_trader.capital,
                'pnl': pnl,
                'pnl_percent': pnl_pct,
                'positions': len(self.ai_trader.positions),
                'day_trades': self.ai_trader.day_trades_count
            })
            
            # Print status every few iterations
            if iteration % 3 == 0 or actions_taken:
                elapsed = (current_time - start_time).total_seconds() / 60
                print(f"[{elapsed:.1f}min] Portfolio: ${portfolio_value:,.2f} ({pnl_pct:+.2f}%) | "
                      f"Trades: {self.ai_trader.day_trades_count}/3 | Positions: {len(self.ai_trader.positions)}")
                
                for action in actions_taken:
                    print(f"  -> {action}")
                
                # Show current prices
                price_display = " | ".join([f"{s}: ${current_prices[s]:.2f}" for s in self.symbols if s in current_prices])
                print(f"  Prices: {price_display}")
            
            # Fast sleep
            time.sleep(self.simulation_speed)
        
        return performance_log
    
    def show_results(self, performance_log):
        """Display final results"""
        if not performance_log:
            print("No performance data to display")
            return
        
        final = performance_log[-1]
        initial = self.ai_trader.initial_capital
        
        print("\n" + "=" * 60)
        print("FAST DAY TRADING SIMULATION RESULTS")
        print("=" * 60)
        print(f"Initial Capital:      ${initial:,.2f}")
        print(f"Final Portfolio:      ${final['portfolio_value']:,.2f}")
        print(f"Total P&L:           ${final['pnl']:,.2f} ({final['pnl_percent']:+.2f}%)")
        print(f"Cash Remaining:       ${final['cash']:,.2f}")
        print(f"Day Trades Used:      {final['day_trades']}/3")
        print(f"Total Iterations:     {final['iteration']}")
        print(f"Trades Executed:      {len(self.ai_trader.trade_history)}")
        
        # Show trade history
        if self.ai_trader.trade_history:
            print(f"\nTrade History:")
            for trade in self.ai_trader.trade_history[-5:]:  # Last 5 trades
                if 'pnl' in trade:
                    print(f"  {trade['timestamp'].strftime('%H:%M:%S')} - "
                          f"{trade['action']} {trade['quantity']} {trade['symbol']} @ "
                          f"${trade['price']:.2f} | P&L: ${trade['pnl']:+.2f}")
                else:
                    print(f"  {trade['timestamp'].strftime('%H:%M:%S')} - "
                          f"{trade['action']} {trade['quantity']} {trade['symbol']} @ "
                          f"${trade['price']:.2f}")
        
        # Plot performance
        if len(performance_log) > 1:
            times = [p['timestamp'] for p in performance_log]
            values = [p['portfolio_value'] for p in performance_log]
            
            plt.figure(figsize=(12, 6))
            plt.plot(times, values, 'b-', linewidth=2, label='Portfolio Value')
            plt.axhline(y=initial, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
            plt.title('Fast Day Trading Performance')
            plt.xlabel('Time')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Example usage - MUCH FASTER
if __name__ == "__main__":
    # Create fast simulation
    sim = FastDayTradingSimulation(['AAPL', 'SPY', 'MSFT'], capital=25000)
    
    # Initialize
    if sim.initialize():
        print("Starting fast demo session...")
        
        # Run 10-minute demo (updates every 5 seconds)
        results = sim.run_fast_session(duration_minutes=10)
        
        # Show results
        sim.show_results(results)
    else:
        print("Failed to initialize simulation")