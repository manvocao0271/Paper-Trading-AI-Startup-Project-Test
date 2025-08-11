import asyncio
import websockets
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
import uuid
import random
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
import keyboard
import os
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Setup logging to file to keep console clean
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AlpacaConfig:
    """Alpaca API configuration"""
    api_key: str = os.environ.get('ALPACA_API_KEY', '')
    api_secret: str = os.environ.get('ALPACA_API_SECRET', '')
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading URL
    data_url: str = "https://data.alpaca.markets"
    websocket_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    paper_trading: bool = True

class MockDataFeed:
    """
    Mock data feed that generates realistic market data for active trading
    This ensures constant trading opportunities without waiting for real market conditions
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.current_prices = {}    
        self.price_history = {}
        self.trade_history = {}
        self.callbacks = []
        self.is_running = False
        
        # Initialize with realistic starting prices
        base_prices = {
            'AAPL': 195.0,
            'MSFT': 420.0,
            'GOOGL': 175.0,
            'TSLA': 250.0,
            'SPY': 550.0
        }
        
        for symbol in symbols:
            start_price = base_prices.get(symbol, 100.0)
            self.current_prices[symbol] = start_price
            self.price_history[symbol] = deque([start_price] * 50, maxlen=1000)
            self.trade_history[symbol] = deque(maxlen=1000)
    
    async def start_feed(self):
        """Start generating mock market data"""
        self.is_running = True
        while self.is_running:
            for symbol in self.symbols:
                # Generate realistic price movement
                current = self.current_prices[symbol]
                volatility = 0.002  # 0.2% volatility per update
                change = random.gauss(0, volatility)
                
                # Add some trending behavior
                if random.random() < 0.1:  # 10% chance of trend
                    change += random.choice([-0.005, 0.005])  # 0.5% trend
                
                new_price = current * (1 + change)
                new_price = max(new_price, current * 0.95)  # Circuit breaker
                new_price = min(new_price, current * 1.05)
                
                self.current_prices[symbol] = new_price
                self.price_history[symbol].append(new_price)
                
                # Create trade data
                trade_data = {
                    'symbol': symbol,
                    'price': new_price,
                    'size': random.randint(100, 1000),
                    'timestamp': datetime.now()
                }
                self.trade_history[symbol].append(trade_data)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        await callback(trade_data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            await asyncio.sleep(0.5)  # Update every 500ms for active trading
    
    def get_current_price(self, symbol: str) -> float:
        return self.current_prices.get(symbol, 0)
    
    def get_price_history(self, symbol: str, length: int = 20) -> List[float]:
        history = list(self.price_history.get(symbol, []))
        return history[-length:] if len(history) >= length else history
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def stop(self):
        self.is_running = False

class Portfolio:
    """Portfolio management for both AI and Human"""
    
    def __init__(self, name: str, initial_cash: float = 100000):
        self.name = name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> quantity
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        
    def get_portfolio_value(self, data_feed) -> float:
        """Calculate total portfolio value"""
        value = self.cash
        for symbol, qty in self.positions.items():
            if qty != 0:
                current_price = data_feed.get_current_price(symbol)
                value += qty * current_price
        return value
    
    def get_pnl(self, data_feed) -> Tuple[float, float]:
        """Get P&L in dollars and percentage"""
        current_value = self.get_portfolio_value(data_feed)
        pnl_dollars = current_value - self.initial_cash
        pnl_percent = (pnl_dollars / self.initial_cash) * 100
        return pnl_dollars, pnl_percent
    
    def can_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if we can afford to buy"""
        cost = quantity * price
        return self.cash >= cost
    
    def can_sell(self, symbol: str, quantity: int) -> bool:
        """Check if we have enough shares to sell"""
        return self.positions.get(symbol, 0) >= quantity
    
    def execute_trade(self, symbol: str, quantity: int, price: float, side: str) -> bool:
        """Execute a trade"""
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

class EnhancedAI:
    """More aggressive AI trader for active competition"""
    
    def __init__(self, portfolio: Portfolio, data_feed):
        self.portfolio = portfolio
        self.data_feed = data_feed
        self.last_trade_time = {}
        self.trade_cooldown = 3  # seconds between trades per symbol
        
    async def analyze_and_trade(self):
        """More aggressive trading analysis"""
        for symbol in self.data_feed.symbols:
            current_time = datetime.now()
            
            # Cooldown check
            if symbol in self.last_trade_time:
                if (current_time - self.last_trade_time[symbol]).seconds < self.trade_cooldown:
                    continue
            
            current_price = self.data_feed.get_current_price(symbol)
            price_history = self.data_feed.get_price_history(symbol, 20)
            
            if len(price_history) < 10:
                continue
            
            # Multiple trading strategies
            signal = self._get_trading_signal(symbol, price_history, current_price)
            
            if signal == 'BUY':
                await self._execute_buy(symbol, current_price)
            elif signal == 'SELL':
                await self._execute_sell(symbol, current_price)
    
    def _get_trading_signal(self, symbol: str, prices: List[float], current_price: float) -> str:
        """Enhanced signal generation with multiple strategies"""
        if len(prices) < 10:
            return 'HOLD'
        
        # Strategy 1: Moving Average Crossover
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-10:])
        ma_signal = 'BUY' if short_ma > long_ma * 1.001 else 'SELL' if short_ma < long_ma * 0.999 else 'HOLD'
        
        # Strategy 2: Mean Reversion
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        mr_signal = 'SELL' if z_score > 1.5 else 'BUY' if z_score < -1.5 else 'HOLD'
        
        # Strategy 3: Momentum
        momentum = (current_price - prices[-10]) / prices[-10]
        mom_signal = 'BUY' if momentum > 0.01 else 'SELL' if momentum < -0.01 else 'HOLD'
        
        # Combine signals (majority vote)
        signals = [ma_signal, mr_signal, mom_signal]
        buy_votes = signals.count('BUY')
        sell_votes = signals.count('SELL')
        
        if buy_votes >= 2:
            return 'BUY'
        elif sell_votes >= 2:
            return 'SELL'
        else:
            return 'HOLD'
    
    async def _execute_buy(self, symbol: str, price: float):
        """Execute AI buy order"""
        # Position sizing: use 10-20% of available cash
        available_cash = self.portfolio.cash
        position_size = random.uniform(0.10, 0.20) * available_cash
        quantity = int(position_size / price)
        
        if quantity > 0 and self.portfolio.can_buy(symbol, quantity, price):
            if self.portfolio.execute_trade(symbol, quantity, price, 'buy'):
                self.last_trade_time[symbol] = datetime.now()
                logger.info(f"AI BUY: {quantity} {symbol} @ ${price:.2f}")
    
    async def _execute_sell(self, symbol: str, price: float):
        """Execute AI sell order"""
        current_position = self.portfolio.positions.get(symbol, 0)
        if current_position > 0:
            # Sell 50-100% of position
            sell_quantity = int(current_position * random.uniform(0.5, 1.0))
            if sell_quantity > 0 and self.portfolio.can_sell(symbol, sell_quantity):
                if self.portfolio.execute_trade(symbol, sell_quantity, price, 'sell'):
                    self.last_trade_time[symbol] = datetime.now()
                    logger.info(f"AI SELL: {sell_quantity} {symbol} @ ${price:.2f}")

class TradingCompetition:
    """Interactive trading competition interface"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_feed = MockDataFeed(symbols)
        
        # Create portfolios
        self.human_portfolio = Portfolio("HUMAN", 100000)
        self.ai_portfolio = Portfolio("AI", 100000)
        
        # Create AI trader
        self.ai_trader = EnhancedAI(self.ai_portfolio, self.data_feed)
        
        # UI
        self.console = Console()
        self.running = True
        self.selected_symbol = 0
        self.trade_quantity = 100
        
        # Competition stats
        self.start_time = datetime.now()
        self.competition_duration = 300  # 5 minutes
        
    def create_display(self) -> Layout:
        """Create the live display layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="controls", size=8)
        )
        
        layout["main"].split_row(
            Layout(name="portfolios"),
            Layout(name="market_data")
        )
        
        return layout
    
    def update_display(self, layout: Layout):
        """Update the live display"""
        # Header
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining = max(0, self.competition_duration - elapsed)
        mins, secs = divmod(int(remaining), 60)
        
        layout["header"].update(
            Panel(
                f"🤖 AI vs HUMAN TRADING COMPETITION 🤖\n"
                f"Time Remaining: {mins:02d}:{secs:02d} | "
                f"Press 'q' to quit, 'h' for help",
                style="bold green"
            )
        )
        
        # Portfolio comparison
        human_value = self.human_portfolio.get_portfolio_value(self.data_feed)
        ai_value = self.ai_portfolio.get_portfolio_value(self.data_feed)
        human_pnl_dollars, human_pnl_percent = self.human_portfolio.get_pnl(self.data_feed)
        ai_pnl_dollars, ai_pnl_percent = self.ai_portfolio.get_pnl(self.data_feed)
        
        portfolio_table = Table(title="Portfolio Comparison", box=box.ROUNDED)
        portfolio_table.add_column("Metric", style="cyan")
        portfolio_table.add_column("Human", style="green")
        portfolio_table.add_column("AI", style="red")
        
        portfolio_table.add_row("Portfolio Value", f"${human_value:,.2f}", f"${ai_value:,.2f}")
        portfolio_table.add_row("P&L ($)", f"${human_pnl_dollars:+,.2f}", f"${ai_pnl_dollars:+,.2f}")
        portfolio_table.add_row("P&L (%)", f"{human_pnl_percent:+.2f}%", f"{ai_pnl_percent:+.2f}%")
        portfolio_table.add_row("Cash", f"${self.human_portfolio.cash:,.2f}", f"${self.ai_portfolio.cash:,.2f}")
        portfolio_table.add_row("Total Trades", str(self.human_portfolio.total_trades), str(self.ai_portfolio.total_trades))
        
        # Winner indicator
        if human_value > ai_value:
            winner = "🏆 HUMAN WINNING! 🏆"
            winner_style = "bold green"
        elif ai_value > human_value:
            winner = "🤖 AI WINNING! 🤖"
            winner_style = "bold red"
        else:
            winner = "🤝 TIE GAME! 🤝"
            winner_style = "bold yellow"
        
        portfolio_panel = Panel(portfolio_table, title=winner, title_align="center", style=winner_style)
        layout["portfolios"].update(portfolio_panel)
        
        # Market data
        market_table = Table(title="Live Market Data", box=box.ROUNDED)
        market_table.add_column("Symbol", style="cyan")
        market_table.add_column("Price", style="white")
        market_table.add_column("Change %", style="white")
        market_table.add_column("Human Pos", style="green")
        market_table.add_column("AI Pos", style="red")
        
        for i, symbol in enumerate(self.symbols):
            current_price = self.data_feed.get_current_price(symbol)
            history = self.data_feed.get_price_history(symbol, 2)
            
            if len(history) >= 2:
                change_pct = ((history[-1] - history[-2]) / history[-2]) * 100
                change_style = "green" if change_pct >= 0 else "red"
                change_str = f"{change_pct:+.2f}%"
            else:
                change_str = "0.00%"
                change_style = "white"
            
            human_pos = self.human_portfolio.positions.get(symbol, 0)
            ai_pos = self.ai_portfolio.positions.get(symbol, 0)
            
            # Highlight selected symbol
            symbol_style = "bold yellow" if i == self.selected_symbol else "cyan"
            
            market_table.add_row(
                f"{'▶ ' if i == self.selected_symbol else '  '}{symbol}",
                f"${current_price:.2f}",
                f"[{change_style}]{change_str}[/{change_style}]",
                str(human_pos),
                str(ai_pos),
                style=symbol_style if i == self.selected_symbol else None
            )
        
        layout["market_data"].update(Panel(market_table))
        
        # Controls
        selected_symbol = self.symbols[self.selected_symbol]
        selected_price = self.data_feed.get_current_price(selected_symbol)
        buy_cost = self.trade_quantity * selected_price
        human_pos = self.human_portfolio.positions.get(selected_symbol, 0)
        
        controls_text = f"""
[bold cyan]TRADING CONTROLS[/bold cyan]
Selected: {selected_symbol} @ ${selected_price:.2f}
Quantity: {self.trade_quantity} shares (Cost: ${buy_cost:,.2f})
Your Position: {human_pos} shares

[yellow]↑/↓[/yellow] Select Symbol  [yellow]←/→[/yellow] Adjust Quantity
[green]B[/green] Buy  [red]S[/red] Sell  [blue]C[/blue] Close Position
[yellow]Q[/yellow] Quit  [yellow]H[/yellow] Help
        """
        
        layout["controls"].update(Panel(controls_text, title="Controls"))
    
    async def handle_input(self):
        """Handle keyboard input"""
        while self.running:
            try:
                if keyboard.is_pressed('q'):
                    self.running = False
                    break
                elif keyboard.is_pressed('up'):
                    self.selected_symbol = (self.selected_symbol - 1) % len(self.symbols)
                    await asyncio.sleep(0.2)
                elif keyboard.is_pressed('down'):
                    self.selected_symbol = (self.selected_symbol + 1) % len(self.symbols)
                    await asyncio.sleep(0.2)
                elif keyboard.is_pressed('left'):
                    self.trade_quantity = max(10, self.trade_quantity - 10)
                    await asyncio.sleep(0.1)
                elif keyboard.is_pressed('right'):
                    self.trade_quantity = min(1000, self.trade_quantity + 10)
                    await asyncio.sleep(0.1)
                elif keyboard.is_pressed('b'):
                    await self.execute_human_trade('buy')
                    await asyncio.sleep(0.3)
                elif keyboard.is_pressed('s'):
                    await self.execute_human_trade('sell')
                    await asyncio.sleep(0.3)
                elif keyboard.is_pressed('c'):
                    await self.close_human_position()
                    await asyncio.sleep(0.3)
                elif keyboard.is_pressed('h'):
                    self.show_help()
                    await asyncio.sleep(0.5)
                
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"Input handling error: {e}")
                await asyncio.sleep(0.1)
    
    async def execute_human_trade(self, side: str):
        """Execute human trade"""
        symbol = self.symbols[self.selected_symbol]
        price = self.data_feed.get_current_price(symbol)
        
        if side == 'buy':
            if self.human_portfolio.can_buy(symbol, self.trade_quantity, price):
                if self.human_portfolio.execute_trade(symbol, self.trade_quantity, price, 'buy'):
                    self.console.print(f"✅ BOUGHT {self.trade_quantity} {symbol} @ ${price:.2f}", style="green")
            else:
                self.console.print("❌ Insufficient cash!", style="red")
        
        elif side == 'sell':
            if self.human_portfolio.can_sell(symbol, self.trade_quantity):
                if self.human_portfolio.execute_trade(symbol, self.trade_quantity, price, 'sell'):
                    self.console.print(f"✅ SOLD {self.trade_quantity} {symbol} @ ${price:.2f}", style="red")
            else:
                self.console.print("❌ Insufficient shares!", style="red")
    
    async def close_human_position(self):
        """Close entire position in selected symbol"""
        symbol = self.symbols[self.selected_symbol]
        position = self.human_portfolio.positions.get(symbol, 0)
        
        if position > 0:
            price = self.data_feed.get_current_price(symbol)
            if self.human_portfolio.execute_trade(symbol, position, price, 'sell'):
                self.console.print(f"✅ CLOSED {position} {symbol} position @ ${price:.2f}", style="yellow")
        else:
            self.console.print("❌ No position to close!", style="red")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
🏆 TRADING COMPETITION HELP 🏆

OBJECTIVE: Beat the AI by achieving higher portfolio value!

CONTROLS:
• ↑/↓ Arrow Keys: Select different stocks
• ←/→ Arrow Keys: Adjust trade quantity (10-1000 shares)
• B: Buy selected stock
• S: Sell selected stock  
• C: Close entire position in selected stock
• Q: Quit competition
• H: Show this help

TIPS:
• Watch for price momentum and trends
• The AI is aggressive - trade actively!
• Manage your cash wisely
• Use the 5-minute time limit strategically
• Green = winning, Red = losing

Good luck beating the AI! 🤖
        """
        self.console.print(Panel(help_text, title="Help", style="blue"))
    
    async def ai_trading_loop(self):
        """AI trading loop"""
        while self.running:
            try:
                await self.ai_trader.analyze_and_trade()
                await asyncio.sleep(2)  # AI trades every 2 seconds
            except Exception as e:
                logger.error(f"AI trading error: {e}")
                await asyncio.sleep(1)
    
    def show_final_results(self):
        """Show final competition results"""
        human_value = self.human_portfolio.get_portfolio_value(self.data_feed)
        ai_value = self.ai_portfolio.get_portfolio_value(self.data_feed)
        human_pnl_dollars, human_pnl_percent = self.human_portfolio.get_pnl(self.data_feed)
        ai_pnl_dollars, ai_pnl_percent = self.ai_portfolio.get_pnl(self.data_feed)
        
        results_table = Table(title="🏆 FINAL RESULTS 🏆", box=box.DOUBLE_EDGE)
        results_table.add_column("Trader", style="cyan")
        results_table.add_column("Final Value", style="white")
        results_table.add_column("P&L ($)", style="white")
        results_table.add_column("P&L (%)", style="white")
        results_table.add_column("Total Trades", style="white")
        results_table.add_column("Result", style="white")
        
        if human_value > ai_value:
            human_result = "🏆 WINNER!"
            ai_result = "❌ LOSER"
            human_style = "bold green"
            ai_style = "dim red"
        elif ai_value > human_value:
            human_result = "❌ LOSER"
            ai_result = "🏆 WINNER!"
            human_style = "dim red"
            ai_style = "bold green"
        else:
            human_result = "🤝 TIE"
            ai_result = "🤝 TIE"
            human_style = "bold yellow"
            ai_style = "bold yellow"
        
        results_table.add_row(
            "HUMAN", f"${human_value:,.2f}", f"${human_pnl_dollars:+,.2f}", 
            f"{human_pnl_percent:+.2f}%", str(self.human_portfolio.total_trades), 
            human_result, style=human_style
        )
        results_table.add_row(
            "AI", f"${ai_value:,.2f}", f"${ai_pnl_dollars:+,.2f}", 
            f"{ai_pnl_percent:+.2f}%", str(self.ai_portfolio.total_trades), 
            ai_result, style=ai_style
        )
        
        self.console.print()
        self.console.print(results_table)
        self.console.print()
        
        # Performance analysis
        human_efficiency = human_pnl_percent / max(self.human_portfolio.total_trades, 1)
        ai_efficiency = ai_pnl_percent / max(self.ai_portfolio.total_trades, 1)
        
        self.console.print(f"Human Trading Efficiency: {human_efficiency:.2f}% per trade")
        self.console.print(f"AI Trading Efficiency: {ai_efficiency:.2f}% per trade")
        self.console.print()
        self.console.print("Thanks for playing! Press any key to exit...")
    
    async def run_competition(self):
        """Run the main competition"""
        layout = self.create_display()
        
        # Start background tasks
        data_task = asyncio.create_task(self.data_feed.start_feed())
        ai_task = asyncio.create_task(self.ai_trading_loop())
        input_task = asyncio.create_task(self.handle_input())
        
        # Main display loop
        try:
            with Live(layout, refresh_per_second=10, screen=True):
                while self.running:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    
                    # Check if competition time is up
                    if elapsed >= self.competition_duration:
                        self.running = False
                        break
                    
                    self.update_display(layout)
                    await asyncio.sleep(0.1)
        
        finally:
            # Clean up
            self.data_feed.stop()
            data_task.cancel()
            ai_task.cancel()
            input_task.cancel()
            
            # Show final results
            self.show_final_results()

# Enhanced main function
def main():
    """Run the interactive trading competition"""
    print("🚀 Starting Interactive AI vs Human Trading Competition...")
    print("📊 This demo uses simulated market data for active trading")
    print("⚠️  Make sure you have 'keyboard' and 'rich' packages installed:")
    print("   pip install keyboard rich")
    print()
    
    try:
        # Check required packages
        import keyboard
        from rich.console import Console
        
        # Trading symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        
        # Create and run competition
        competition = TradingCompetition(symbols)
        asyncio.run(competition.run_competition())
        
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install with: pip install keyboard rich")
    except KeyboardInterrupt:
        print("\n👋 Competition ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()