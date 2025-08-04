import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from collections import deque
import time
import json

class MockDataFeed:
    """Enhanced data feed with more realistic market behavior"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.current_prices = {}
        self.price_history = {}
        self.volume_history = {}
        self.news_events = {}
        self.round_number = 0
        self.market_sentiment = 0.0  # -1 to 1 (bearish to bullish)
        
        # Enhanced starting prices with different sectors
        base_prices = {
            'AAPL': 195.0,   # Tech
            'MSFT': 420.0,   # Tech
            'GOOGL': 175.0,  # Tech
            'TSLA': 250.0,   # Auto/Energy
            'SPY': 550.0,    # Index
            'NVDA': 880.0,   # AI/Tech
            'JPM': 180.0,    # Finance
            'JNJ': 165.0,    # Healthcare
            'XOM': 110.0,    # Energy
            'WMT': 165.0     # Retail
        }
        
        self.sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
            'TSLA': 'Automotive', 'JPM': 'Finance', 'JNJ': 'Healthcare', 
            'XOM': 'Energy', 'WMT': 'Retail', 'SPY': 'Index'
        }
        
        for symbol in symbols:
            start_price = base_prices.get(symbol, 100.0)
            self.current_prices[symbol] = start_price
            self.price_history[symbol] = deque([start_price] * 10, maxlen=100)
            self.volume_history[symbol] = deque([1000000] * 10, maxlen=100)
            self.news_events[symbol] = []
    
    def generate_market_event(self):
        """Generate random market events that affect prices"""
        events = []
        
        if random.random() < 0.15:  # 15% chance of market-wide event
            event_type = random.choice(['bullish', 'bearish', 'neutral'])
            if event_type == 'bullish':
                self.market_sentiment = min(1.0, self.market_sentiment + 0.3)
                events.append("📈 MARKET NEWS: Positive economic data boosts investor confidence")
            elif event_type == 'bearish':
                self.market_sentiment = max(-1.0, self.market_sentiment - 0.3)
                events.append("📉 MARKET NEWS: Economic concerns weigh on markets")
            else:
                events.append("📊 MARKET NEWS: Mixed signals as investors await key data")
        
        # Stock-specific events
        for symbol in self.symbols:
            if random.random() < 0.08:  # 8% chance per stock
                sector = self.sector_map.get(symbol, 'General')
                event_templates = {
                    'Technology': [
                        f"💻 {symbol}: Strong quarterly earnings beat expectations",
                        f"🚀 {symbol}: New product launch generates buzz",
                        f"⚠️ {symbol}: Regulatory concerns in key markets"
                    ],
                    'Finance': [
                        f"🏦 {symbol}: Interest rate changes affect outlook",
                        f"📊 {symbol}: Credit quality improvements noted",
                        f"⚖️ {symbol}: Regulatory review announced"
                    ],
                    'Healthcare': [
                        f"💊 {symbol}: FDA approval for new treatment",
                        f"🔬 {symbol}: Promising clinical trial results",
                        f"⚠️ {symbol}: Patent expiration concerns"
                    ],
                    'Energy': [
                        f"⛽ {symbol}: Oil price volatility impacts outlook",
                        f"🌱 {symbol}: Green energy investments announced",
                        f"📉 {symbol}: Production costs rising"
                    ]
                }
                
                templates = event_templates.get(sector, [f"📰 {symbol}: Company news affects trading"])
                event = random.choice(templates)
                events.append(event)
                self.news_events[symbol].append(event)
        
        return events
    
    def advance_round(self):
        """Generate new prices with enhanced market dynamics"""
        self.round_number += 1
        
        # Generate market events
        events = self.generate_market_event()
        
        for symbol in self.symbols:
            current = self.current_prices[symbol]
            
            # Base volatility varies by stock type
            base_volatility = {
                'TSLA': 0.025, 'NVDA': 0.025,  # High volatility
                'AAPL': 0.018, 'MSFT': 0.016, 'GOOGL': 0.020,  # Medium-high
                'JPM': 0.015, 'JNJ': 0.012, 'WMT': 0.013,  # Medium
                'XOM': 0.020, 'SPY': 0.012  # Varies
            }.get(symbol, 0.015)
            
            # Market sentiment influence
            sentiment_effect = self.market_sentiment * 0.01
            
            # Generate price change
            change = random.gauss(sentiment_effect, base_volatility)
            
            # Add momentum and mean reversion
            history = list(self.price_history[symbol])
            if len(history) >= 5:
                recent_trend = (history[-1] - history[-5]) / history[-5]
                # Momentum (trend continuation)
                if abs(recent_trend) > 0.02:
                    momentum = recent_trend * 0.3
                    change += momentum
                
                # Mean reversion (occasional reversals)
                if random.random() < 0.15:
                    change -= recent_trend * 0.5
            
            # Stock-specific news impact
            if symbol in [event.split(':')[0].split()[-1] for event in events if ':' in event]:
                news_impact = random.gauss(0, 0.015)
                change += news_impact
            
            # Apply change with circuit breakers
            new_price = current * (1 + change)
            new_price = max(new_price, current * 0.85)  # 15% down limit
            new_price = min(new_price, current * 1.15)  # 15% up limit
            
            self.current_prices[symbol] = new_price
            self.price_history[symbol].append(new_price)
            
            # Generate volume (higher volume with bigger price moves)
            base_volume = 1000000
            volume_multiplier = 1 + abs(change) * 10
            volume = int(base_volume * volume_multiplier * random.uniform(0.5, 2.0))
            self.volume_history[symbol].append(volume)
        
        # Decay market sentiment
        self.market_sentiment *= 0.9
        
        return events
    
    def get_current_price(self, symbol: str) -> float:
        return self.current_prices.get(symbol, 0)
    
    def get_price_change(self, symbol: str) -> Tuple[float, float]:
        """Get price change from previous round"""
        history = list(self.price_history.get(symbol, []))
        if len(history) >= 2:
            current = history[-1]
            previous = history[-2]
            change_dollars = current - previous
            change_percent = (change_dollars / previous) * 100
            return change_dollars, change_percent
        return 0.0, 0.0
    
    def get_technical_analysis(self, symbol: str) -> Dict:
        """Enhanced technical analysis"""
        history = list(self.price_history.get(symbol, []))
        volumes = list(self.volume_history.get(symbol, []))
        
        if len(history) < 10:
            return {"trend": "INSUFFICIENT_DATA", "rsi": 50, "support": 0, "resistance": 0}
        
        # Moving averages
        ma_5 = np.mean(history[-5:])
        ma_10 = np.mean(history[-10:])
        ma_20 = np.mean(history[-20:]) if len(history) >= 20 else ma_10
        
        # RSI calculation (simplified)
        changes = np.diff(history[-14:]) if len(history) >= 14 else np.diff(history)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.01
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Support and resistance
        recent_prices = history[-20:] if len(history) >= 20 else history
        support = min(recent_prices)
        resistance = max(recent_prices)
        
        # Trend determination
        if ma_5 > ma_10 * 1.02 and ma_10 > ma_20 * 1.01:
            trend = "STRONG_UPTREND"
        elif ma_5 > ma_10 * 1.005:
            trend = "UPTREND"
        elif ma_5 < ma_10 * 0.98 and ma_10 < ma_20 * 0.99:
            trend = "STRONG_DOWNTREND"
        elif ma_5 < ma_10 * 0.995:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        return {
            "trend": trend,
            "rsi": rsi,
            "support": support,
            "resistance": resistance,
            "ma_5": ma_5,
            "ma_10": ma_10,
            "volume": volumes[-1] if volumes else 0
        }

class Portfolio:
    """Enhanced portfolio with performance metrics"""
    
    def __init__(self, name: str, initial_cash: float = 100000):
        self.name = name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_values = []
        
    def get_portfolio_value(self, data_feed) -> float:
        """Calculate total portfolio value"""
        value = self.cash
        for symbol, qty in self.positions.items():
            if qty != 0:
                current_price = data_feed.get_current_price(symbol)
                value += qty * current_price
        return value
    
    def get_position_value(self, symbol: str, data_feed) -> float:
        """Get value of specific position"""
        qty = self.positions.get(symbol, 0)
        if qty == 0:
            return 0
        return qty * data_feed.get_current_price(symbol)
    
    def get_position_pnl(self, symbol: str, data_feed) -> Tuple[float, float]:
        """Get P&L for specific position"""
        qty = self.positions.get(symbol, 0)
        if qty == 0:
            return 0, 0
        
        current_price = data_feed.get_current_price(symbol)
        current_value = qty * current_price
        
        # Calculate average cost basis from trade history
        total_cost = 0
        total_shares = 0
        for trade in self.trade_history:
            if trade['symbol'] == symbol:
                if trade['side'] == 'BUY':
                    total_cost += trade['value']
                    total_shares += trade['quantity']
                else:  # SELL
                    if total_shares > 0:
                        avg_cost = total_cost / total_shares
                        sold_cost = trade['quantity'] * avg_cost
                        total_cost -= sold_cost
                        total_shares -= trade['quantity']
        
        if total_shares > 0 and total_cost > 0:
            cost_basis = total_cost
            pnl_dollars = current_value - cost_basis
            pnl_percent = (pnl_dollars / cost_basis) * 100
            return pnl_dollars, pnl_percent
        
        return 0, 0
    
    def get_pnl(self, data_feed) -> Tuple[float, float]:
        """Get total portfolio P&L"""
        current_value = self.get_portfolio_value(data_feed)
        pnl_dollars = current_value - self.initial_cash
        pnl_percent = (pnl_dollars / self.initial_cash) * 100
        return pnl_dollars, pnl_percent
    
    def get_performance_metrics(self, data_feed) -> Dict:
        """Get detailed performance metrics"""
        current_value = self.get_portfolio_value(data_feed)
        total_pnl = current_value - self.initial_cash
        total_return = (total_pnl / self.initial_cash) * 100
        
        # Win rate
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        # Sharpe ratio (simplified)
        if len(self.daily_values) > 1:
            returns = np.diff(self.daily_values) / self.daily_values[:-1]
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            sharpe = (avg_return / max(return_std, 0.001)) * np.sqrt(252) if return_std > 0 else 0
        else:
            sharpe = 0
        
        return {
            "total_return": total_return,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "total_trades": self.total_trades,
            "cash_ratio": (self.cash / current_value) * 100
        }
    
    def execute_trade(self, symbol: str, quantity: int, price: float, side: str) -> bool:
        """Execute a trade with enhanced tracking"""
        if side.lower() == 'buy':
            cost = quantity * price
            if self.cash >= cost:
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                self.total_trades += 1
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'value': cost
                })
                return True
        
        elif side.lower() == 'sell':
            if self.positions.get(symbol, 0) >= quantity:
                revenue = quantity * price
                self.cash += revenue
                self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                self.total_trades += 1
                
                # Check if this was a winning trade
                cost_basis = self.get_avg_cost_basis(symbol)
                if cost_basis > 0 and price > cost_basis:
                    self.winning_trades += 1
                
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'value': revenue
                })
                return True
        
        return False
    
    def get_avg_cost_basis(self, symbol: str) -> float:
        """Calculate average cost basis for a symbol"""
        total_cost = 0
        total_shares = 0
        
        for trade in self.trade_history:
            if trade['symbol'] == symbol and trade['side'] == 'BUY':
                total_cost += trade['value']
                total_shares += trade['quantity']
        
        return total_cost / total_shares if total_shares > 0 else 0

class EnhancedAI:
    """Advanced AI with multiple strategies and adaptive behavior"""
    
    def __init__(self, portfolio: Portfolio, data_feed):
        self.portfolio = portfolio
        self.data_feed = data_feed
        self.strategy_mode = "ADAPTIVE"  # MOMENTUM, MEAN_REVERSION, ADAPTIVE
        self.risk_tolerance = 0.7  # 0-1 scale
        self.recent_performance = deque(maxlen=10)
        
    def analyze_market_regime(self) -> str:
        """Determine current market regime"""
        spy_history = list(self.data_feed.price_history.get('SPY', []))
        if len(spy_history) < 10:
            return "UNCERTAIN"
        
        recent_volatility = np.std(spy_history[-10:]) / np.mean(spy_history[-10:])
        recent_trend = (spy_history[-1] - spy_history[-10]) / spy_history[-10]
        
        if recent_volatility > 0.15:
            return "HIGH_VOLATILITY"
        elif recent_trend > 0.05:
            return "BULL_MARKET"
        elif recent_trend < -0.05:
            return "BEAR_MARKET"
        else:
            return "SIDEWAYS_MARKET"
    
    def get_stock_score(self, symbol: str) -> Tuple[float, List[str]]:
        """Score a stock based on multiple factors"""
        analysis = self.data_feed.get_technical_analysis(symbol)
        current_price = self.data_feed.get_current_price(symbol)
        
        score = 0
        reasons = []
        
        # Technical score
        if analysis["trend"] == "STRONG_UPTREND":
            score += 3
            reasons.append("Strong uptrend confirmed")
        elif analysis["trend"] == "UPTREND":
            score += 1.5
            reasons.append("Uptrend in progress")
        elif analysis["trend"] == "STRONG_DOWNTREND":
            score -= 3
            reasons.append("Strong downtrend warning")
        elif analysis["trend"] == "DOWNTREND":
            score -= 1.5
            reasons.append("Downtrend detected")
        
        # RSI analysis
        rsi = analysis.get("rsi", 50)
        if rsi < 30:
            score += 2
            reasons.append(f"Oversold (RSI: {rsi:.1f})")
        elif rsi > 70:
            score -= 2
            reasons.append(f"Overbought (RSI: {rsi:.1f})")
        elif 45 <= rsi <= 55:
            score += 0.5
            reasons.append("Neutral RSI momentum")
        
        # Support/Resistance analysis
        support = analysis.get("support", 0)
        resistance = analysis.get("resistance", 0)
        if support > 0 and resistance > 0:
            distance_to_support = (current_price - support) / support
            distance_to_resistance = (resistance - current_price) / current_price
            
            if distance_to_support < 0.02:
                score += 1.5
                reasons.append("Near support level")
            elif distance_to_resistance < 0.02:
                score -= 1.5
                reasons.append("Near resistance level")
        
        # Volume analysis
        volume_history = list(self.data_feed.volume_history.get(symbol, []))
        if len(volume_history) >= 2:
            volume_ratio = volume_history[-1] / volume_history[-2]
            if volume_ratio > 1.5:
                score += 1
                reasons.append("High volume surge")
            elif volume_ratio < 0.7:
                score -= 0.5
                reasons.append("Low volume concern")
        
        # Market sentiment
        if self.data_feed.market_sentiment > 0.3:
            score += 1
            reasons.append("Positive market sentiment")
        elif self.data_feed.market_sentiment < -0.3:
            score -= 1
            reasons.append("Negative market sentiment")
        
        return score, reasons
    
    def get_position_management_signal(self, symbol: str) -> Tuple[str, List[str]]:
        """Determine what to do with existing positions"""
        position = self.portfolio.positions.get(symbol, 0)
        if position == 0:
            return "NO_POSITION", []
        
        pnl_dollars, pnl_percent = self.portfolio.get_position_pnl(symbol, self.data_feed)
        reasons = []
        
        # Stop loss (7% loss)
        if pnl_percent < -7:
            return "STOP_LOSS", [f"Position down {pnl_percent:.1f}% - cutting losses"]
        
        # Take profit (15% gain)
        if pnl_percent > 15:
            return "TAKE_PROFIT", [f"Position up {pnl_percent:.1f}% - securing profits"]
        
        # Trailing stop logic
        score, score_reasons = self.get_stock_score(symbol)
        if pnl_percent > 5 and score < -2:
            return "TRAILING_STOP", [f"Profitable position but negative signals: {', '.join(score_reasons[:2])}"]
        
        # Add to winners
        if pnl_percent > 3 and score > 2:
            return "ADD_TO_WINNER", [f"Position profitable and strong signals: {', '.join(score_reasons[:2])}"]
        
        return "HOLD_POSITION", [f"Current P&L: {pnl_percent:+.1f}%"]
    
    def get_planned_action(self) -> Tuple[str, str, int, str]:
        """Get AI's planned action with sophisticated analysis"""
        market_regime = self.analyze_market_regime()
        
        # Adjust strategy based on market regime
        if market_regime == "HIGH_VOLATILITY":
            self.risk_tolerance = 0.4
        elif market_regime == "BULL_MARKET":
            self.risk_tolerance = 0.8
        elif market_regime == "BEAR_MARKET":
            self.risk_tolerance = 0.3
        else:
            self.risk_tolerance = 0.6
        
        best_action = "HOLD"
        best_symbol = ""
        best_quantity = 0
        explanation = f"Market Regime: {market_regime}. "
        
        # First, check existing positions
        position_actions = []
        for symbol in self.portfolio.positions:
            if self.portfolio.positions[symbol] > 0:
                action, reasons = self.get_position_management_signal(symbol)
                if action != "HOLD_POSITION":
                    position_actions.append((symbol, action, reasons))
        
        # Execute most critical position management first
        critical_actions = ["STOP_LOSS", "TAKE_PROFIT"]
        for symbol, action, reasons in position_actions:
            if action in critical_actions:
                quantity = self.portfolio.positions[symbol]
                explanation += f"{action.replace('_', ' ')} {symbol}: {reasons[0]}"
                return action, symbol, quantity, explanation
        
        # Check for new opportunities
        stock_scores = []
        for symbol in self.data_feed.symbols:
            score, reasons = self.get_stock_score(symbol)
            stock_scores.append((symbol, score, reasons))
        
        # Sort by score
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Look for buy opportunities
        for symbol, score, reasons in stock_scores[:3]:  # Top 3 stocks
            if score > 2.5:  # Strong buy signal
                # Position sizing based on conviction and risk
                max_position_size = self.portfolio.cash * (self.risk_tolerance * 0.25)  # Max 25% of cash
                current_price = self.data_feed.get_current_price(symbol)
                max_shares = int(max_position_size / current_price)
                
                # Scale position size by conviction
                conviction_multiplier = min(score / 5.0, 1.0)
                quantity = max(10, int(max_shares * conviction_multiplier))
                quantity = min(quantity, 500)  # Cap at 500 shares
                
                if quantity * current_price <= self.portfolio.cash:
                    explanation += f"BUY opportunity in {symbol} (Score: {score:.1f}): {', '.join(reasons[:3])}"
                    return "BUY", symbol, quantity, explanation
        
        # Look for adding to winners
        for symbol, action, reasons in position_actions:
            if action == "ADD_TO_WINNER":
                max_add = self.portfolio.cash * 0.1  # Add max 10% of cash
                current_price = self.data_feed.get_current_price(symbol)
                quantity = max(10, int(max_add / current_price))
                quantity = min(quantity, 200)
                
                if quantity * current_price <= self.portfolio.cash:
                    explanation += f"Adding to winning position {symbol}: {reasons[0]}"
                    return "ADD", symbol, quantity, explanation
        
        # Look for sells (beyond stop losses)
        for symbol, score, reasons in stock_scores[-3:]:  # Bottom 3 stocks
            if score < -2.5 and self.portfolio.positions.get(symbol, 0) > 0:
                quantity = self.portfolio.positions[symbol]
                explanation += f"SELL {symbol} due to weak signals (Score: {score:.1f}): {', '.join(reasons[:2])}"
                return "SELL", symbol, quantity, explanation
        
        explanation += "No strong signals detected. Maintaining current positions and cash reserves."
        return "HOLD", "", 0, explanation
    
    def execute_planned_action(self, action: str, symbol: str, quantity: int) -> bool:
        """Execute the planned action"""
        if action == "HOLD" or quantity <= 0:
            return True
        
        price = self.data_feed.get_current_price(symbol)
        
        if action in ["BUY", "ADD", "ADD_TO_WINNER"]:
            return self.portfolio.execute_trade(symbol, quantity, price, 'buy')
        elif action in ["SELL", "STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP"]:
            return self.portfolio.execute_trade(symbol, quantity, price, 'sell')
        
        return False

class EnhancedTradingGUI:
    """Enhanced GUI with better visuals and more features"""
    
    def __init__(self):
        # Enhanced symbol list
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'NVDA', 'JPM', 'JNJ', 'XOM', 'WMT']
        self.data_feed = MockDataFeed(self.symbols)
        
        # Create portfolios
        self.human_portfolio = Portfolio("HUMAN", 100000)
        self.ai_portfolio = Portfolio("AI", 100000)
        
        # Create enhanced AI
        self.ai = EnhancedAI(self.ai_portfolio, self.data_feed)
        
        # Game state
        self.round_number = 0
        self.max_rounds = 25  # More rounds for deeper strategy
        self.current_phase = "MARKET_UPDATE"
        self.market_events = []
        self.game_over = False
        
        # AI state
        self.ai_planned_action = None
        self.ai_planned_symbol = None
        self.ai_planned_quantity = 0
        self.ai_explanation = ""
        
        # Performance tracking
        self.performance_history = {"human": [], "ai": []}
        
        # Setup GUI
        self.setup_gui()
        
        # Generate initial market events to populate the news
        self.market_events = [
            "🎯 Welcome to the Elite Trading Competition!",
            "📊 Market is open and ready for trading",
            "💡 Use technical analysis to your advantage"
        ]
        
        # Initial display update
        self.update_display()
        self.update_action_section()
    
    def setup_gui(self):
        """Setup enhanced GUI"""
        self.root = tk.Tk()
        self.root.title("🎯 Elite Trading Competition - AI vs Human")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1a1a1a')
        
        # Enhanced styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom color scheme
        colors = {
            'bg': '#1a1a1a',
            'fg': '#ffffff',
            'accent': '#00ff88',
            'warning': '#ff6b6b',
            'info': '#4ecdc4',
            'gold': '#ffd700'
        }
        
        style.configure('Title.TLabel', font=('Arial', 20, 'bold'), 
                       background=colors['bg'], foreground=colors['gold'])
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'), 
                       background=colors['bg'], foreground=colors['accent'])
        style.configure('Data.TLabel', font=('Arial', 11), 
                       background=colors['bg'], foreground=colors['fg'])
        style.configure('Win.TLabel', font=('Arial', 12, 'bold'), 
                       background=colors['bg'], foreground=colors['accent'])
        style.configure('Lose.TLabel', font=('Arial', 12, 'bold'), 
                       background=colors['bg'], foreground=colors['warning'])
        
        self.create_enhanced_widgets()
    
    def create_enhanced_widgets(self):
        """Create enhanced widget layout"""
        # Main container with padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        self.create_enhanced_header(main_frame)
        
        # Three-column layout
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        
        # Left column - Market & News
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Middle column - Action & Strategy
        middle_frame = ttk.Frame(content_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Right column - Portfolio & Performance
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create sections
        self.create_market_section(left_frame)
        self.create_news_section(left_frame)
        
        self.create_action_section(middle_frame)
        self.create_insights_section(middle_frame)
        
        self.create_portfolio_section(right_frame)
        self.create_performance_section(right_frame)
    
    def create_enhanced_header(self, parent):
        """Create enhanced header with game status"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(fill=tk.X)
        
        self.title_label = ttk.Label(title_frame, text="🎯 ELITE TRADING COMPETITION", style='Title.TLabel')
        self.title_label.pack(side=tk.LEFT)
        
        # Status indicators
        status_frame = ttk.Frame(title_frame)
        status_frame.pack(side=tk.RIGHT)
        
        self.round_label = ttk.Label(status_frame, text="Round 0/25", style='Header.TLabel')
        self.round_label.pack(side=tk.RIGHT, padx=(20, 0))
        
        self.phase_label = ttk.Label(status_frame, text="Phase: Market Update", style='Header.TLabel')
        self.phase_label.pack(side=tk.RIGHT, padx=(20, 0))
        
        # Market regime indicator
        self.market_regime_label = ttk.Label(status_frame, text="Market: Analyzing...", style='Data.TLabel')
        self.market_regime_label.pack(side=tk.RIGHT, padx=(20, 0))
    
    def create_market_section(self, parent):
        """Create enhanced market data section"""
        market_frame = ttk.LabelFrame(parent, text="📊 LIVE MARKET DATA", padding=15)
        market_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Market data table with enhanced columns
        columns = ('Symbol', 'Price', 'Change', '%', 'Volume', 'RSI', 'Trend', 'Your Pos', 'AI Pos')
        self.market_tree = ttk.Treeview(market_frame, columns=columns, show='headings', height=8)
        
        # Configure columns with appropriate widths
        widths = [70, 90, 80, 60, 80, 50, 90, 70, 70]
        for col, width in zip(columns, widths):
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=width, anchor='center')
        
        # Add scrollbar
        scrollbar_market = ttk.Scrollbar(market_frame, orient=tk.VERTICAL, command=self.market_tree.yview)
        self.market_tree.configure(yscrollcommand=scrollbar_market.set)
        
        self.market_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_market.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize with symbols
        for symbol in self.symbols:
            self.market_tree.insert('', 'end', iid=symbol)
        
        # Market summary
        summary_frame = ttk.Frame(market_frame)
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.market_summary_label = ttk.Label(summary_frame, text="Market Summary: Initializing...", 
                                            style='Data.TLabel')
        self.market_summary_label.pack()
    
    def create_news_section(self, parent):
        """Create news and events section"""
        news_frame = ttk.LabelFrame(parent, text="📰 MARKET NEWS & EVENTS", padding=15)
        news_frame.pack(fill=tk.BOTH, expand=True)
        
        # News display area
        self.news_text = tk.Text(news_frame, height=6, bg='#2d2d2d', fg='#ffffff', 
                               font=('Arial', 10), wrap=tk.WORD, state='disabled')
        news_scrollbar = ttk.Scrollbar(news_frame, orient=tk.VERTICAL, command=self.news_text.yview)
        self.news_text.configure(yscrollcommand=news_scrollbar.set)
        
        self.news_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        news_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_action_section(self, parent):
        """Create dynamic action section"""
        self.action_frame = ttk.LabelFrame(parent, text="🎮 GAME CONTROL", padding=15)
        self.action_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.action_content = ttk.Frame(self.action_frame)
        self.action_content.pack(fill=tk.BOTH, expand=True)
    
    def create_insights_section(self, parent):
        """Create strategy insights section"""
        insights_frame = ttk.LabelFrame(parent, text="🧠 STRATEGIC ANALYSIS", padding=15)
        insights_frame.pack(fill=tk.BOTH, expand=True)
        
        # Insights text area
        self.insights_text = tk.Text(insights_frame, height=8, bg='#1a1a1a', fg='#00ff88', 
                                   font=('Consolas', 10), wrap=tk.WORD, state='disabled')
        insights_scrollbar = ttk.Scrollbar(insights_frame, orient=tk.VERTICAL, command=self.insights_text.yview)
        self.insights_text.configure(yscrollcommand=insights_scrollbar.set)
        
        self.insights_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        insights_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_portfolio_section(self, parent):
        """Create enhanced portfolio comparison"""
        portfolio_frame = ttk.LabelFrame(parent, text="💰 PORTFOLIO BATTLE", padding=15)
        portfolio_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Human portfolio
        human_frame = ttk.Frame(portfolio_frame)
        human_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(human_frame, text="👤 YOU", style='Header.TLabel').pack()
        self.human_value_label = ttk.Label(human_frame, text="$100,000", style='Data.TLabel')
        self.human_value_label.pack(pady=2)
        self.human_pnl_label = ttk.Label(human_frame, text="±$0 (0.00%)", style='Data.TLabel')
        self.human_pnl_label.pack(pady=2)
        self.human_cash_label = ttk.Label(human_frame, text="Cash: $100,000", style='Data.TLabel')
        self.human_cash_label.pack(pady=2)
        self.human_trades_label = ttk.Label(human_frame, text="Trades: 0", style='Data.TLabel')
        self.human_trades_label.pack(pady=2)
        
        # VS separator with winner indicator
        vs_frame = ttk.Frame(portfolio_frame)
        vs_frame.pack(side=tk.LEFT, padx=20)
        
        self.winner_label = ttk.Label(vs_frame, text="VS", font=('Arial', 18, 'bold'),
                                     background='#1a1a1a', foreground='#ffd700')
        self.winner_label.pack()
        
        self.lead_label = ttk.Label(vs_frame, text="", style='Data.TLabel')
        self.lead_label.pack(pady=(5, 0))
        
        # AI portfolio
        ai_frame = ttk.Frame(portfolio_frame)
        ai_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(ai_frame, text="🤖 AI", style='Header.TLabel').pack()
        self.ai_value_label = ttk.Label(ai_frame, text="$100,000", style='Data.TLabel')
        self.ai_value_label.pack(pady=2)
        self.ai_pnl_label = ttk.Label(ai_frame, text="±$0 (0.00%)", style='Data.TLabel')
        self.ai_pnl_label.pack(pady=2)
        self.ai_cash_label = ttk.Label(ai_frame, text="Cash: $100,000", style='Data.TLabel')
        self.ai_cash_label.pack(pady=2)
        self.ai_trades_label = ttk.Label(ai_frame, text="Trades: 0", style='Data.TLabel')
        self.ai_trades_label.pack(pady=2)
    
    def create_performance_section(self, parent):
        """Create performance metrics section"""
        perf_frame = ttk.LabelFrame(parent, text="📈 PERFORMANCE METRICS", padding=15)
        perf_frame.pack(fill=tk.BOTH, expand=True)
        
        # Performance comparison table
        perf_columns = ('Metric', 'You', 'AI', 'Winner')
        self.perf_tree = ttk.Treeview(perf_frame, columns=perf_columns, show='headings', height=6)
        
        for col in perf_columns:
            self.perf_tree.heading(col, text=col)
            self.perf_tree.column(col, width=80, anchor='center')
        
        self.perf_tree.pack(fill=tk.BOTH, expand=True)
        
        # Initialize performance metrics
        metrics = ['Total Return', 'Win Rate', 'Sharpe Ratio', 'Total Trades', 'Cash %']
        for i, metric in enumerate(metrics):
            self.perf_tree.insert('', 'end', iid=f'metric_{i}', values=(metric, '-', '-', '-'))
    
    def update_action_section(self):
        """Update action section based on current phase"""
        # Clear existing content
        for widget in self.action_content.winfo_children():
            widget.destroy()
        
        if self.current_phase == "MARKET_UPDATE":
            self.create_market_update_actions()
        elif self.current_phase == "HUMAN_TURN":
            self.create_human_turn_actions()
        elif self.current_phase == "AI_PREVIEW":
            self.create_ai_preview_actions()
        elif self.current_phase == "AI_EXECUTE":
            self.create_ai_execute_actions()
    
    def create_market_update_actions(self):
        """Market update phase actions"""
        self.action_frame.configure(text="📈 MARKET UPDATE")
        
        # Welcome message for first round
        if self.round_number == 0:
            welcome_frame = ttk.LabelFrame(self.action_content, text="Welcome to Elite Trading!", padding=15)
            welcome_frame.pack(fill=tk.X, pady=(0, 15))
            
            welcome_text = """🎯 COMPETITION OVERVIEW:
• 25 rounds of strategic trading
• $100,000 starting capital
• Beat the AI across multiple metrics
• Real-time market simulation

📊 FEATURES:
• Technical analysis tools
• Market news & events  
• AI strategy previews
• Performance tracking"""
            
            ttk.Label(welcome_frame, text=welcome_text, style='Data.TLabel', justify='left').pack(anchor='w')
        
        # Show market events if any
        if len(self.market_events) > 3:  # Skip initial welcome messages
            events_frame = ttk.LabelFrame(self.action_content, text="Latest Market News", padding=10)
            events_frame.pack(fill=tk.X, pady=(0, 15))
            
            recent_events = self.market_events[-3:]  # Show last 3 events
            events_text = "\n".join(recent_events)
            ttk.Label(events_frame, text=events_text, style='Data.TLabel', justify='left').pack(anchor='w')
        
        # Market status
        status_text = "Market data updated. Ready to begin trading!" if self.round_number == 0 else "Market data updated for this round."
        ttk.Label(self.action_content, text=status_text, style='Data.TLabel').pack(pady=10)
        
        # Start trading button
        start_btn = ttk.Button(self.action_content, text="▶️ START TRADING", 
                             command=self.advance_to_human_turn,
                             style='Accent.TButton')
        start_btn.pack(pady=15)
        
        # Add custom button style
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 12, 'bold'))
    
    def create_human_turn_actions(self):
        """Enhanced human turn actions"""
        self.action_frame.configure(text="🎯 YOUR MOVE")
        
        # Quick market overview
        overview_frame = ttk.Frame(self.action_content)
        overview_frame.pack(fill=tk.X, pady=(0, 15))
        
        market_regime = self.ai.analyze_market_regime()
        ttk.Label(overview_frame, text=f"Market Regime: {market_regime}", style='Header.TLabel').pack()
        
        # Stock selection with enhanced info
        selection_frame = ttk.LabelFrame(self.action_content, text="Stock Selection", padding=10)
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Stock selector
        selector_frame = ttk.Frame(selection_frame)
        selector_frame.pack(fill=tk.X)
        
        ttk.Label(selector_frame, text="Stock:", style='Data.TLabel').pack(side=tk.LEFT)
        self.selected_symbol = tk.StringVar(value=self.symbols[0])
        symbol_combo = ttk.Combobox(selector_frame, textvariable=self.selected_symbol,
                                   values=self.symbols, state='readonly', width=8)
        symbol_combo.pack(side=tk.LEFT, padx=(5, 15))
        symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_selected)
        
        ttk.Label(selector_frame, text="Quantity:", style='Data.TLabel').pack(side=tk.LEFT)
        self.trade_quantity = tk.IntVar(value=100)
        quantity_spin = tk.Spinbox(selector_frame, from_=10, to=1000, increment=10,
                                  textvariable=self.trade_quantity, width=8)
        quantity_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Stock analysis display
        self.stock_analysis_label = ttk.Label(selection_frame, text="", style='Data.TLabel', justify='left')
        self.stock_analysis_label.pack(fill=tk.X, pady=(10, 0))
        
        # Position info
        self.position_info_label = ttk.Label(selection_frame, text="", style='Data.TLabel')
        self.position_info_label.pack(fill=tk.X, pady=(5, 0))
        
        # Action buttons
        button_frame = ttk.Frame(self.action_content)
        button_frame.pack(pady=15)
        
        # Buy/Sell buttons
        trade_frame = ttk.Frame(button_frame)
        trade_frame.pack(pady=(0, 10))
        
        buy_btn = ttk.Button(trade_frame, text="🟢 BUY", command=self.human_buy)
        buy_btn.pack(side=tk.LEFT, padx=5)
        
        sell_btn = ttk.Button(trade_frame, text="🔴 SELL", command=self.human_sell)
        sell_btn.pack(side=tk.LEFT, padx=5)
        
        # Management buttons
        mgmt_frame = ttk.Frame(button_frame)
        mgmt_frame.pack()
        
        close_btn = ttk.Button(mgmt_frame, text="🚪 CLOSE POSITION", command=self.human_close)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        hold_btn = ttk.Button(mgmt_frame, text="⏸️ HOLD", command=self.human_hold)
        hold_btn.pack(side=tk.LEFT, padx=5)
        
        self.update_trade_info()
    
    def create_ai_preview_actions(self):
        """Enhanced AI preview phase"""
        self.action_frame.configure(text="🤖 AI STRATEGY PREVIEW")
        
        # AI strategy display
        strategy_frame = ttk.LabelFrame(self.action_content, text="AI's Master Plan", padding=15)
        strategy_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Action summary
        action_text = f"🎯 Planned Action: {self.ai_planned_action}\n"
        if self.ai_planned_symbol:
            current_price = self.data_feed.get_current_price(self.ai_planned_symbol)
            estimated_cost = self.ai_planned_quantity * current_price
            action_text += f"📊 Target: {self.ai_planned_symbol}\n"
            action_text += f"📈 Quantity: {self.ai_planned_quantity:,} shares\n"
            action_text += f"💰 Estimated Value: ${estimated_cost:,.2f}\n"
        
        ttk.Label(strategy_frame, text=action_text, style='Data.TLabel', justify='left').pack(anchor='w')
        
        # AI reasoning
        reasoning_frame = ttk.LabelFrame(self.action_content, text="AI's Reasoning", padding=15)
        reasoning_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(reasoning_frame, text=self.ai_explanation, style='Data.TLabel', 
                 justify='left', wraplength=400).pack(anchor='w')
        
        # Execute button
        execute_btn = ttk.Button(self.action_content, text="▶️ LET AI EXECUTE", 
                               command=self.advance_to_ai_execute)
        execute_btn.pack(pady=10)
    
    def create_ai_execute_actions(self):
        """AI execution phase"""
        self.action_frame.configure(text="🤖 AI EXECUTING")
        
        # Execution result
        if self.ai_planned_action == "HOLD":
            result_text = "🤖 AI Analysis Complete: No trades executed this round.\nAI is maintaining current positions and waiting for better opportunities."
        else:
            symbol = self.ai_planned_symbol
            quantity = self.ai_planned_quantity
            price = self.data_feed.get_current_price(symbol)
            cost = quantity * price
            
            success = self.ai.execute_planned_action(self.ai_planned_action, symbol, quantity)
            
            if success:
                action_verb = "BOUGHT" if self.ai_planned_action in ["BUY", "ADD"] else "SOLD"
                result_text = f"✅ AI {action_verb}: {quantity:,} shares of {symbol}\n"
                result_text += f"💰 Execution Price: ${price:.2f}\n"
                result_text += f"💵 Total Value: ${cost:,.2f}\n"
                result_text += f"🎯 Strategy: {self.ai_planned_action.replace('_', ' ')}"
            else:
                result_text = f"❌ AI execution failed: {self.ai_planned_action} {quantity} {symbol}\n"
                result_text += "Insufficient funds or position size."
        
        result_frame = ttk.LabelFrame(self.action_content, text="Execution Result", padding=15)
        result_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(result_frame, text=result_text, style='Data.TLabel', justify='left').pack(anchor='w')
        
        # Next action
        if self.round_number >= self.max_rounds:
            finish_btn = ttk.Button(self.action_content, text="🏁 FINISH GAME", 
                                  command=self.end_game)
            finish_btn.pack(pady=15)
        else:
            next_btn = ttk.Button(self.action_content, text="▶️ NEXT ROUND", 
                                command=self.next_round)
            next_btn.pack(pady=15)

    def on_symbol_selected(self, event=None):
        """Handle symbol selection with enhanced info"""
        self.update_trade_info()
    
    def update_trade_info(self):
        """Update trade information with technical analysis"""
        if not hasattr(self, 'selected_symbol'):
            return
            
        symbol = self.selected_symbol.get()
        quantity = self.trade_quantity.get()
        price = self.data_feed.get_current_price(symbol)
        cost = quantity * price
        
        # Get technical analysis
        analysis = self.data_feed.get_technical_analysis(symbol)
        change_dollars, change_percent = self.data_feed.get_price_change(symbol)
        
        # Position info
        position = self.human_portfolio.positions.get(symbol, 0)
        position_value = position * price if position > 0 else 0
        
        # Stock analysis text
        analysis_text = f"📊 {symbol} Analysis:\n"
        analysis_text += f"Price: ${price:.2f} ({change_dollars:+.2f} / {change_percent:+.2f}%)\n"
        analysis_text += f"Trend: {analysis['trend'].replace('_', ' ')} | RSI: {analysis['rsi']:.1f}\n"
        analysis_text += f"Support: ${analysis['support']:.2f} | Resistance: ${analysis['resistance']:.2f}"
        
        # Position info text
        if position > 0:
            pnl_dollars, pnl_percent = self.human_portfolio.get_position_pnl(symbol, self.data_feed)
            position_text = f"Your Position: {position:,} shares (${position_value:,.2f})\n"
            position_text += f"P&L: ${pnl_dollars:+,.2f} ({pnl_percent:+.2f}%)"
        else:
            position_text = f"No current position in {symbol}\n"
            position_text += f"Trade Cost: ${cost:,.2f} | Available Cash: ${self.human_portfolio.cash:,.2f}"
        
        if hasattr(self, 'stock_analysis_label'):
            self.stock_analysis_label.configure(text=analysis_text)
        if hasattr(self, 'position_info_label'):
            self.position_info_label.configure(text=position_text)
    
    def human_buy(self):
        """Execute human buy with validation"""
        symbol = self.selected_symbol.get()
        quantity = self.trade_quantity.get()
        price = self.data_feed.get_current_price(symbol)
        cost = quantity * price
        
        if self.human_portfolio.cash >= cost:
            if self.human_portfolio.execute_trade(symbol, quantity, price, 'buy'):
                self.log_insight(f"✅ YOU BOUGHT: {quantity:,} {symbol} @ ${price:.2f} (${cost:,.2f})")
                analysis = self.data_feed.get_technical_analysis(symbol)
                self.log_insight(f"📊 Technical: {analysis['trend'].replace('_', ' ')}, RSI: {analysis['rsi']:.1f}")
                self.advance_to_ai_preview()
            else:
                self.log_insight("❌ Buy order failed")
        else:
            shortage = cost - self.human_portfolio.cash
            self.log_insight(f"❌ Insufficient funds! Need ${shortage:,.2f} more cash")
    
    def human_sell(self):
        """Execute human sell with validation"""
        symbol = self.selected_symbol.get()
        quantity = self.trade_quantity.get()
        price = self.data_feed.get_current_price(symbol)
        position = self.human_portfolio.positions.get(symbol, 0)
        
        if position >= quantity:
            if self.human_portfolio.execute_trade(symbol, quantity, price, 'sell'):
                revenue = quantity * price
                pnl_dollars, pnl_percent = self.human_portfolio.get_position_pnl(symbol, self.data_feed)
                self.log_insight(f"✅ YOU SOLD: {quantity:,} {symbol} @ ${price:.2f} (${revenue:,.2f})")
                self.log_insight(f"💰 Trade P&L: ${pnl_dollars:+,.2f} ({pnl_percent:+.2f}%)")
                self.advance_to_ai_preview()
            else:
                self.log_insight("❌ Sell order failed")
        else:
            self.log_insight(f"❌ Insufficient shares! You have {position:,}, trying to sell {quantity:,}")
    
    def human_close(self):
        """Close entire human position"""
        symbol = self.selected_symbol.get()
        position = self.human_portfolio.positions.get(symbol, 0)
        
        if position > 0:
            price = self.data_feed.get_current_price(symbol)
            pnl_dollars, pnl_percent = self.human_portfolio.get_position_pnl(symbol, self.data_feed)
            
            if self.human_portfolio.execute_trade(symbol, position, price, 'sell'):
                revenue = position * price
                self.log_insight(f"✅ POSITION CLOSED: {position:,} {symbol} @ ${price:.2f}")
                self.log_insight(f"💰 Final P&L: ${pnl_dollars:+,.2f} ({pnl_percent:+.2f}%) | Revenue: ${revenue:,.2f}")
                self.advance_to_ai_preview()
            else:
                self.log_insight("❌ Close position failed")
        else:
            self.log_insight(f"❌ No position to close in {symbol}")
    
    def human_hold(self):
        """Human chooses to hold"""
        self.log_insight("⏸️ YOU CHOSE TO HOLD - Maintaining current positions")
        current_value = self.human_portfolio.get_portfolio_value(self.data_feed)
        cash_ratio = (self.human_portfolio.cash / current_value) * 100
        self.log_insight(f"💼 Portfolio: ${current_value:,.2f} | Cash: {cash_ratio:.1f}%")
        self.advance_to_ai_preview()
    
    def advance_to_human_turn(self):
        """Advance to human turn"""
        self.current_phase = "HUMAN_TURN"
        self.update_action_section()
        
        # Market analysis for the player
        market_regime = self.ai.analyze_market_regime()
        self.log_insight(f"🎯 ROUND {self.round_number + 1} - YOUR TURN")
        self.log_insight(f"📊 Market Regime: {market_regime}")
        
        # Show top opportunities
        stock_scores = []
        for symbol in self.symbols:
            score, reasons = self.ai.get_stock_score(symbol)
            stock_scores.append((symbol, score, reasons))
        
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        top_stock = stock_scores[0]
        if top_stock[1] > 1:
            self.log_insight(f"💡 AI HINT: {top_stock[0]} showing strength - {top_stock[2][0] if top_stock[2] else 'Strong signals'}")
    
    def advance_to_ai_preview(self):
        """Advance to AI preview"""
        self.current_phase = "AI_PREVIEW"
        
        # Get AI's planned action
        self.ai_planned_action, self.ai_planned_symbol, self.ai_planned_quantity, self.ai_explanation = self.ai.get_planned_action()
        
        self.update_action_section()
        self.log_insight(f"🤖 AI STRATEGY: {self.ai_explanation}")
    
    def advance_to_ai_execute(self):
        """Advance to AI execution"""
        self.current_phase = "AI_EXECUTE"
        self.update_action_section()
    
    def next_round(self):
        """Start next round"""
        self.round_number += 1
        self.current_phase = "MARKET_UPDATE"
        
        # Generate new market data and events
        new_events = self.data_feed.advance_round()
        self.market_events.extend(new_events)
        
        # Update performance tracking
        human_value = self.human_portfolio.get_portfolio_value(self.data_feed)
        ai_value = self.ai_portfolio.get_portfolio_value(self.data_feed)
        self.performance_history["human"].append(human_value)
        self.performance_history["ai"].append(ai_value)
        
        # Update portfolios' daily values for Sharpe calculation
        self.human_portfolio.daily_values.append(human_value)
        self.ai_portfolio.daily_values.append(ai_value)
        
        self.update_action_section()
        self.update_display()
        
        self.log_insight(f"📊 ROUND {self.round_number} BEGINS")
        if new_events:
            for event in new_events[-2:]:  # Show last 2 events
                self.log_insight(f"📰 {event}")
    
    def update_display(self):
        """Enhanced display update"""
        if self.game_over:
            return
        
        # Update header
        self.round_label.configure(text=f"Round {self.round_number}/{self.max_rounds}")
        
        phase_names = {
            "MARKET_UPDATE": "📈 Market Update",
            "HUMAN_TURN": "👤 Your Turn",
            "AI_PREVIEW": "🤖 AI Preview", 
            "AI_EXECUTE": "🤖 AI Execute"
        }
        self.phase_label.configure(text=f"Phase: {phase_names.get(self.current_phase, self.current_phase)}")
        
        # Update market regime
        market_regime = self.ai.analyze_market_regime()
        regime_colors = {
            "BULL_MARKET": "#00ff00",
            "BEAR_MARKET": "#ff4444", 
            "HIGH_VOLATILITY": "#ff8800",
            "SIDEWAYS_MARKET": "#ffff00",
            "UNCERTAIN": "#888888"
        }
        color = regime_colors.get(market_regime, "#ffffff")
        self.market_regime_label.configure(text=f"Market: {market_regime.replace('_', ' ')}", foreground=color)
        
        # Update market data
        total_volume = 0
        advancing = 0
        declining = 0
        
        for symbol in self.symbols:
            current_price = self.data_feed.get_current_price(symbol)
            change_dollars, change_percent = self.data_feed.get_price_change(symbol)
            analysis = self.data_feed.get_technical_analysis(symbol)
            
            human_pos = self.human_portfolio.positions.get(symbol, 0)
            ai_pos = self.ai_portfolio.positions.get(symbol, 0)
            
            # Format volume in millions
            volume = analysis.get('volume', 0)
            volume_str = f"{volume/1000000:.1f}M" if volume >= 1000000 else f"{volume/1000:.0f}K"
            total_volume += volume
            
            # Count advancing/declining
            if change_percent > 0:
                advancing += 1
            elif change_percent < 0:
                declining += 1
            
            # Update tree
            self.market_tree.set(symbol, 'Price', f"${current_price:.2f}")
            self.market_tree.set(symbol, 'Change', f"${change_dollars:+.2f}")
            self.market_tree.set(symbol, '%', f"{change_percent:+.2f}%")
            self.market_tree.set(symbol, 'Volume', volume_str)
            self.market_tree.set(symbol, 'RSI', f"{analysis.get('rsi', 50):.0f}")
            self.market_tree.set(symbol, 'Trend', analysis.get('trend', 'SIDEWAYS').replace('_', ' '))
            self.market_tree.set(symbol, 'Your Pos', f"{human_pos:,}" if human_pos > 0 else "-")
            self.market_tree.set(symbol, 'AI Pos', f"{ai_pos:,}" if ai_pos > 0 else "-")
        
        # Update market summary
        breadth = f"Advancing: {advancing} | Declining: {declining} | Unchanged: {len(self.symbols) - advancing - declining}"
        volume_summary = f"Total Volume: {total_volume/1000000:.1f}M"
        sentiment = f"Sentiment: {self.data_feed.market_sentiment:+.2f}"
        market_summary = f"{breadth} | {volume_summary} | {sentiment}"
        self.market_summary_label.configure(text=market_summary)
        
        # Update portfolios
        human_value = self.human_portfolio.get_portfolio_value(self.data_feed)
        ai_value = self.ai_portfolio.get_portfolio_value(self.data_feed)
        human_pnl_dollars, human_pnl_percent = self.human_portfolio.get_pnl(self.data_feed)
        ai_pnl_dollars, ai_pnl_percent = self.ai_portfolio.get_pnl(self.data_feed)
        
        # Portfolio values
        self.human_value_label.configure(text=f"${human_value:,.2f}")
        self.ai_value_label.configure(text=f"${ai_value:,.2f}")
        
        # P&L with color coding
        human_pnl_color = "#00ff00" if human_pnl_dollars >= 0 else "#ff4444"
        ai_pnl_color = "#00ff00" if ai_pnl_dollars >= 0 else "#ff4444"
        self.human_pnl_label.configure(text=f"${human_pnl_dollars:+,.2f} ({human_pnl_percent:+.2f}%)", 
                                      foreground=human_pnl_color)
        self.ai_pnl_label.configure(text=f"${ai_pnl_dollars:+,.2f} ({ai_pnl_percent:+.2f}%)", 
                                   foreground=ai_pnl_color)
        
        # Cash and trades
        self.human_cash_label.configure(text=f"Cash: ${self.human_portfolio.cash:,.2f}")
        self.ai_cash_label.configure(text=f"Cash: ${self.ai_portfolio.cash:,.2f}")
        self.human_trades_label.configure(text=f"Trades: {self.human_portfolio.total_trades}")
        self.ai_trades_label.configure(text=f"Trades: {self.ai_portfolio.total_trades}")
        
        # Winner indicator
        lead_amount = abs(human_value - ai_value)
        if human_value > ai_value:
            self.winner_label.configure(text="👤 YOU LEAD!", foreground='#00ff00')
            self.lead_label.configure(text=f"By ${lead_amount:,.2f}")
        elif ai_value > human_value:
            self.winner_label.configure(text="🤖 AI LEADS!", foreground='#ff4444')
            self.lead_label.configure(text=f"By ${lead_amount:,.2f}")
        else:
            self.winner_label.configure(text="🤝 TIED!", foreground='#ffff00')
            self.lead_label.configure(text="Perfect tie!")
        
        # Update performance metrics
        human_metrics = self.human_portfolio.get_performance_metrics(self.data_feed)
        ai_metrics = self.ai_portfolio.get_performance_metrics(self.data_feed)
        
        metrics_data = [
            ('Total Return', f"{human_metrics['total_return']:+.2f}%", f"{ai_metrics['total_return']:+.2f}%"),
            ('Win Rate', f"{human_metrics['win_rate']:.1f}%", f"{ai_metrics['win_rate']:.1f}%"),
            ('Sharpe Ratio', f"{human_metrics['sharpe_ratio']:.2f}", f"{ai_metrics['sharpe_ratio']:.2f}"),
            ('Total Trades', f"{human_metrics['total_trades']}", f"{ai_metrics['total_trades']}"),
            ('Cash %', f"{human_metrics['cash_ratio']:.1f}%", f"{ai_metrics['cash_ratio']:.1f}%")
        ]
        
        for i, (metric, human_val, ai_val) in enumerate(metrics_data):
            # Determine winner for this metric
            try:
                human_num = float(human_val.replace('%', ''))
                ai_num = float(ai_val.replace('%', ''))
                if human_num > ai_num:
                    winner = "👤"
                elif ai_num > human_num:
                    winner = "🤖"
                else:
                    winner = "🤝"
            except:
                winner = "-"
            
            self.perf_tree.set(f'metric_{i}', 'Metric', metric)
            self.perf_tree.set(f'metric_{i}', 'You', human_val)
            self.perf_tree.set(f'metric_{i}', 'AI', ai_val)
            self.perf_tree.set(f'metric_{i}', 'Winner', winner)
        
        # Update news section
        if self.market_events:
            self.news_text.configure(state='normal')
            self.news_text.delete(1.0, tk.END)
            
            # Show recent news (last 10 events)
            recent_events = self.market_events[-10:] if len(self.market_events) > 10 else self.market_events
            for event in recent_events:
                timestamp = datetime.now().strftime("%H:%M")
                self.news_text.insert(tk.END, f"[{timestamp}] {event}\n")
            
            self.news_text.see(tk.END)
            self.news_text.configure(state='disabled')
        
        # Update trade info if in human turn
        if self.current_phase == "HUMAN_TURN":
            self.update_trade_info()
    
    def log_insight(self, message: str):
        """Enhanced logging with timestamps and formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.insights_text.configure(state='normal')
        self.insights_text.insert(tk.END, log_message)
        self.insights_text.see(tk.END)
        self.insights_text.configure(state='disabled')
    
    def end_game(self):
        """Enhanced game ending with detailed analysis"""
        self.game_over = True
        
        # Calculate comprehensive final results
        human_value = self.human_portfolio.get_portfolio_value(self.data_feed)
        ai_value = self.ai_portfolio.get_portfolio_value(self.data_feed)
        human_metrics = self.human_portfolio.get_performance_metrics(self.data_feed)
        ai_metrics = self.ai_portfolio.get_performance_metrics(self.data_feed)
        
        # Determine overall winner
        human_score = 0
        ai_score = 0
        
        # Score based on multiple factors
        if human_value > ai_value:
            human_score += 3
        elif ai_value > human_value:
            ai_score += 3
        
        if human_metrics['total_return'] > ai_metrics['total_return']:
            human_score += 2
        elif ai_metrics['total_return'] > human_metrics['total_return']:
            ai_score += 2
        
        if human_metrics['sharpe_ratio'] > ai_metrics['sharpe_ratio']:
            human_score += 1
        elif ai_metrics['sharpe_ratio'] > human_metrics['sharpe_ratio']:
            ai_score += 1
        
        if human_metrics['win_rate'] > ai_metrics['win_rate']:
            human_score += 1
        elif ai_metrics['win_rate'] > human_metrics['win_rate']:
            ai_score += 1
        
        # Final winner determination
        if human_score > ai_score:
            winner = "🏆 CONGRATULATIONS! YOU WIN! 🏆"
            winner_detail = f"You outperformed the AI across multiple metrics!"
            color = "#00ff00"
        elif ai_score > human_score:
            winner = "🤖 AI VICTORY! 🤖"
            winner_detail = f"The AI demonstrated superior trading performance!"
            color = "#ff4444"
        else:
            winner = "🤝 INCREDIBLE TIE! 🤝"
            winner_detail = f"You matched the AI's performance exactly!"
            color = "#ffff00"
        
        # Detailed analysis
        performance_gap = abs(human_value - ai_value)
        return_gap = abs(human_metrics['total_return'] - ai_metrics['total_return'])
        
        result_message = f"""
{winner}
{winner_detail}

📊 FINAL SCORECARD:

👤 YOUR PERFORMANCE:
• Portfolio Value: ${human_value:,.2f}
• Total Return: {human_metrics['total_return']:+.2f}%
• Total P&L: ${human_metrics['total_pnl']:+,.2f}
• Win Rate: {human_metrics['win_rate']:.1f}%
• Sharpe Ratio: {human_metrics['sharpe_ratio']:.2f}
• Total Trades: {human_metrics['total_trades']}
• Cash Remaining: ${self.human_portfolio.cash:,.2f}

🤖 AI PERFORMANCE:
• Portfolio Value: ${ai_value:,.2f}
• Total Return: {ai_metrics['total_return']:+.2f}%
• Total P&L: ${ai_metrics['total_pnl']:+,.2f}
• Win Rate: {ai_metrics['win_rate']:.1f}%
• Sharpe Ratio: {ai_metrics['sharpe_ratio']:.2f}
• Total Trades: {ai_metrics['total_trades']}
• Cash Remaining: ${self.ai_portfolio.cash:,.2f}

🎯 BATTLE STATISTICS:
• Rounds Played: {self.round_number}
• Portfolio Gap: ${performance_gap:,.2f}
• Return Gap: {return_gap:.2f}%
• Your Score: {human_score} points
• AI Score: {ai_score} points

🏅 ACHIEVEMENT ANALYSIS:
• Trading Frequency: {"Active Trader" if human_metrics['total_trades'] > 15 else "Conservative Trader"}
• Risk Management: {"Excellent" if human_metrics['sharpe_ratio'] > 1.0 else "Good" if human_metrics['sharpe_ratio'] > 0.5 else "Needs Work"}
• Market Timing: {"Superb" if human_metrics['win_rate'] > 70 else "Good" if human_metrics['win_rate'] > 50 else "Practice More"}

Thank you for playing the Elite Trading Competition! 🎉
        """
        
        messagebox.showinfo("🏁 GAME COMPLETE!", result_message)
        
        # Update final status
        self.log_insight(f"🏁 FINAL RESULT: {winner}")
        self.log_insight(winner_detail)
        self.log_insight(f"📊 Your Score: {human_score} | AI Score: {ai_score}")
        self.log_insight("🎉 Thanks for playing! What an incredible battle!")
        
        # Update action section for game end
        for widget in self.action_content.winfo_children():
            widget.destroy()
        
        self.action_frame.configure(text="🏁 GAME COMPLETE")
        
        # Winner announcement
        winner_frame = ttk.Frame(self.action_content)
        winner_frame.pack(pady=20)
        
        final_label = ttk.Label(winner_frame, text=winner, 
                               font=('Arial', 16, 'bold'), background='#1a1a1a', foreground=color)
        final_label.pack()
        
        detail_label = ttk.Label(winner_frame, text=winner_detail, style='Data.TLabel')
        detail_label.pack(pady=(10, 0))
        
        # Score display
        score_label = ttk.Label(winner_frame, text=f"Final Score - You: {human_score} | AI: {ai_score}", 
                               style='Header.TLabel')
        score_label.pack(pady=(10, 0))
        
        # Action buttons
        button_frame = ttk.Frame(self.action_content)
        button_frame.pack(pady=20)
        
        play_again_btn = ttk.Button(button_frame, text="🔄 PLAY AGAIN", command=self.restart_game)
        play_again_btn.pack(side=tk.LEFT, padx=10)
        
        save_results_btn = ttk.Button(button_frame, text="💾 SAVE RESULTS", command=self.save_game_results)
        save_results_btn.pack(side=tk.LEFT, padx=10)
        
        quit_btn = ttk.Button(button_frame, text="❌ QUIT", command=self.root.quit)
        quit_btn.pack(side=tk.LEFT, padx=10)
    
    def save_game_results(self):
        """Save game results to file"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'rounds': self.round_number,
                'human': {
                    'final_value': self.human_portfolio.get_portfolio_value(self.data_feed),
                    'metrics': self.human_portfolio.get_performance_metrics(self.data_feed),
                    'trades': len(self.human_portfolio.trade_history)
                },
                'ai': {
                    'final_value': self.ai_portfolio.get_portfolio_value(self.data_feed),
                    'metrics': self.ai_portfolio.get_performance_metrics(self.data_feed),
                    'trades': len(self.ai_portfolio.trade_history)
                },
                'performance_history': self.performance_history
            }
            
            filename = f"trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            messagebox.showinfo("💾 Results Saved", f"Game results saved to {filename}")
            self.log_insight(f"💾 Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("❌ Save Error", f"Failed to save results: {e}")
    
    def restart_game(self):
        """Enhanced game restart"""
        # Reset portfolios
        self.human_portfolio = Portfolio("HUMAN", 100000)
        self.ai_portfolio = Portfolio("AI", 100000)
        
        # Reset AI
        self.ai = EnhancedAI(self.ai_portfolio, self.data_feed)
        
        # Reset game state
        self.round_number = 0
        self.current_phase = "MARKET_UPDATE"
        self.market_events = [
            "🎯 New Elite Trading Competition Started!",
            "📊 Market reset and ready for trading",
            "💡 Good luck beating the AI this time!"
        ]
        self.game_over = False
        self.performance_history = {"human": [], "ai": []}
        
        # Reset data feed
        self.data_feed = MockDataFeed(self.symbols)
        
        # Clear text areas
        self.insights_text.configure(state='normal')
        self.insights_text.delete(1.0, tk.END)
        self.insights_text.configure(state='disabled')
        
        self.news_text.configure(state='normal')
        self.news_text.delete(1.0, tk.END)
        self.news_text.configure(state='disabled')
        
        # Reset AI planned action
        self.ai_planned_action = None
        self.ai_planned_symbol = None
        self.ai_planned_quantity = 0
        self.ai_explanation = ""
        
        # Update display
        self.update_action_section()
        self.update_display()
        
        # Welcome message
        self.log_insight("🎮 NEW ELITE TRADING COMPETITION STARTED!")
        self.log_insight("🎯 25 rounds of strategic trading await!")
        self.log_insight("🧠 The AI has been reset and is ready for battle!")
        self.log_insight("💡 Use technical analysis and market insights to your advantage!")
    
    def run(self):
        """Start the enhanced application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Enhanced startup messages
        self.log_insight("🎯 ELITE TRADING COMPETITION INITIALIZED!")
        self.log_insight("=" * 50)
        self.log_insight("📋 COMPETITION RULES:")
        self.log_insight("• 25 strategic rounds of turn-based trading")
        self.log_insight("• Start with $100,000 virtual cash")
        self.log_insight("• Trade 10 different stocks across sectors") 
        self.log_insight("• AI shows strategy before executing")
        self.log_insight("• Win by achieving higher portfolio value")
        self.log_insight("")
        self.log_insight("🎮 NEW FEATURES:")
        self.log_insight("• Enhanced technical analysis (RSI, Support/Resistance)")
        self.log_insight("• Real-time market news and events")
        self.log_insight("• Advanced AI with multiple strategies")
        self.log_insight("• Comprehensive performance tracking")
        self.log_insight("• Market regime detection")
        self.log_insight("")
        self.log_insight("💡 STRATEGIC TIPS:")
        self.log_insight("• Watch for market regime changes")
        self.log_insight("• Use technical indicators for timing")
        self.log_insight("• Consider risk management and position sizing")
        self.log_insight("• Learn from AI's strategy previews")
        self.log_insight("=" * 50)
        self.log_insight("🚀 READY TO BEGIN! Click 'START TRADING' when ready.")
        
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Exit the Elite Trading Competition?"):
            self.root.destroy()

# Enhanced main function
def main():
    """Run the enhanced elite trading competition"""
    print("🎯 ELITE TRADING COMPETITION")
    print("=" * 50)
    print("🎮 ENHANCED FEATURES:")
    print("  ✅ 10 stocks across multiple sectors")
    print("  ✅ Advanced technical analysis (RSI, Support/Resistance)")
    print("  ✅ Real-time market news and events")
    print("  ✅ Intelligent AI with adaptive strategies")
    print("  ✅ Comprehensive performance metrics")
    print("  ✅ Market regime detection")
    print("  ✅ Enhanced UI with 3-column layout")
    print("  ✅ Game result saving")
    print("")
    print("🧠 AI CAPABILITIES:")
    print("  • Multi-factor stock scoring")
    print("  • Dynamic risk management")
    print("  • Market regime adaptation")
    print("  • Position management strategies")
    print("  • Technical analysis integration")
    print("")
    print("🏆 YOUR MISSION:")
    print("  Beat the AI across 25 strategic rounds!")
    print("  Maximize portfolio value through smart trading!")
    print("=" * 50)
    
    try:
        # Create and run the enhanced competition
        competition = EnhancedTradingGUI()
        competition.run()
        
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install: pip install tkinter numpy")
        print("Note: tkinter usually comes with Python")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Try restarting the application")

if __name__ == "__main__":
    main()