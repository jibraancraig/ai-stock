#!/usr/bin/env python3
"""
Stock Market AI Bot
A comprehensive command-line AI bot for stock analysis and trading recommendations
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse

@dataclass
class StockAnalysis:
    """Data class for stock analysis results"""
    symbol: str
    current_price: float
    price_change: float
    rsi: float
    macd: float
    recommendation: str
    confidence: float
    risk_level: str
    volatility: float
    signals: List[str]
    timestamp: datetime

class StockMarketBot:
    """Main AI Bot for stock market analysis"""
    
    def __init__(self):
        self.analysis_history = []
        self.watchlist = []
        self.load_config()
    
    def load_config(self):
        """Load bot configuration"""
        self.config = {
            'default_period': '6mo',
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'confidence_threshold': 30,
            'max_watchlist': 10
        }
    
    def fetch_stock_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            print(f"📊 Fetching data for {symbol.upper()}...")
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(period=period)
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            
            if len(data) == 0:
                raise ValueError(f"No data found for {symbol}")
            
            print(f"✅ Successfully fetched {len(data)} data points")
            return data
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        try:
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            
            # Moving Averages
            df['MA_5'] = ta.trend.SMAIndicator(df['Close'], window=5).sma_indicator()
            df['MA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['MA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Middle'] = bb.bollinger_mavg()
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Volatility
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            
            return df
        except Exception as e:
            print(f"❌ Error adding indicators: {str(e)}")
            return df
    
    def analyze_stock(self, symbol: str, period: str = "6mo") -> Optional[StockAnalysis]:
        """Perform comprehensive stock analysis"""
        try:
            # Fetch data
            df = self.fetch_stock_data(symbol, period)
            if df is None:
                return None
            
            # Add indicators
            df = self.add_technical_indicators(df)
            
            # Get latest data
            latest = df.iloc[-1]
            prev_20 = df.iloc[-20] if len(df) >= 20 else df.iloc[0]
            
            # Calculate metrics
            current_price = latest['Close']
            price_change = ((current_price - prev_20['Close']) / prev_20['Close']) * 100
            rsi = latest.get('RSI', 50)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            volatility = latest.get('Volatility', 0)
            volume_ratio = latest.get('Volume_Ratio', 1)
            
            # Generate signals
            signals = []
            confidence = 0
            
            # Price trend signals
            if price_change > 10:
                signals.append(f"Strong uptrend (+{price_change:.1f}%)")
                confidence += 30
            elif price_change > 5:
                signals.append(f"Moderate uptrend (+{price_change:.1f}%)")
                confidence += 20
            elif price_change > 0:
                signals.append(f"Slight uptrend (+{price_change:.1f}%)")
                confidence += 10
            elif price_change < -10:
                signals.append(f"Strong downtrend ({price_change:.1f}%)")
                confidence -= 30
            elif price_change < -5:
                signals.append(f"Moderate downtrend ({price_change:.1f}%)")
                confidence -= 20
            else:
                signals.append(f"Sideways movement ({price_change:.1f}%)")
                confidence += 5
            
            # RSI signals
            if rsi < self.config['rsi_oversold']:
                signals.append(f"Oversold (RSI: {rsi:.1f})")
                confidence += 25
            elif rsi > self.config['rsi_overbought']:
                signals.append(f"Overbought (RSI: {rsi:.1f})")
                confidence -= 25
            
            # MACD signals
            if macd > macd_signal:
                signals.append("MACD bullish crossover")
                confidence += 15
            else:
                signals.append("MACD bearish")
                confidence -= 15
            
            # Volume signals
            if volume_ratio > 1.5:
                signals.append(f"High volume ({volume_ratio:.1f}x average)")
                confidence += 10
            elif volume_ratio < 0.5:
                signals.append(f"Low volume ({volume_ratio:.1f}x average)")
                confidence -= 5
            
            # Moving average signals
            if current_price > latest.get('MA_20', current_price):
                signals.append("Price above 20-day MA")
                confidence += 10
            else:
                signals.append("Price below 20-day MA")
                confidence -= 10
            
            # Determine recommendation
            if confidence > self.config['confidence_threshold']:
                recommendation = "BUY"
            elif confidence < -self.config['confidence_threshold']:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            # Risk assessment
            if volatility > 40:
                risk_level = "Very High"
            elif volatility > 30:
                risk_level = "High"
            elif volatility > 20:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            return StockAnalysis(
                symbol=symbol.upper(),
                current_price=current_price,
                price_change=price_change,
                rsi=rsi,
                macd=macd,
                recommendation=recommendation,
                confidence=abs(confidence),
                risk_level=risk_level,
                volatility=volatility,
                signals=signals,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            return None
    
    def display_analysis(self, analysis: StockAnalysis):
        """Display analysis results in a formatted way"""
        print("\n" + "="*60)
        print(f"📊 STOCK ANALYSIS: {analysis.symbol}")
        print("="*60)
        
        # Price information
        print(f"💰 Current Price: ${analysis.current_price:.2f}")
        print(f"📈 20-Day Change: {analysis.price_change:+.2f}%")
        print(f"📊 RSI: {analysis.rsi:.1f}")
        print(f"📊 MACD: {analysis.macd:.3f}")
        print(f"⚠️  Volatility: {analysis.volatility:.1f}%")
        print(f"⚠️  Risk Level: {analysis.risk_level}")
        
        # Recommendation
        print("\n🎯 RECOMMENDATION:")
        if analysis.recommendation == "BUY":
            print(f"🟢 {analysis.recommendation} - Confidence: {analysis.confidence:.0f}%")
        elif analysis.recommendation == "SELL":
            print(f"🔴 {analysis.recommendation} - Confidence: {analysis.confidence:.0f}%")
        else:
            print(f"🟡 {analysis.recommendation} - Confidence: {analysis.confidence:.0f}%")
        
        # Signals
        print("\n📡 KEY SIGNALS:")
        for signal in analysis.signals:
            print(f"   • {signal}")
        
        print("="*60)
    
    def add_to_watchlist(self, symbol: str):
        """Add stock to watchlist"""
        if len(self.watchlist) >= self.config['max_watchlist']:
            print(f"⚠️  Watchlist is full (max {self.config['max_watchlist']} stocks)")
            return
        
        if symbol.upper() not in [s.upper() for s in self.watchlist]:
            self.watchlist.append(symbol.upper())
            print(f"✅ Added {symbol.upper()} to watchlist")
        else:
            print(f"⚠️  {symbol.upper()} is already in watchlist")
    
    def remove_from_watchlist(self, symbol: str):
        """Remove stock from watchlist"""
        if symbol.upper() in [s.upper() for s in self.watchlist]:
            self.watchlist.remove(symbol.upper())
            print(f"✅ Removed {symbol.upper()} from watchlist")
        else:
            print(f"⚠️  {symbol.upper()} not found in watchlist")
    
    def analyze_watchlist(self):
        """Analyze all stocks in watchlist"""
        if not self.watchlist:
            print("📋 Watchlist is empty")
            return
        
        print(f"\n📋 ANALYZING WATCHLIST ({len(self.watchlist)} stocks)")
        print("="*60)
        
        for symbol in self.watchlist:
            analysis = self.analyze_stock(symbol)
            if analysis:
                self.display_analysis(analysis)
                self.analysis_history.append(analysis)
                time.sleep(1)  # Avoid rate limiting
    
    def show_menu(self):
        """Display main menu"""
        print("\n🤖 STOCK MARKET AI BOT")
        print("="*40)
        print("1. 📊 Analyze Single Stock")
        print("2. 📋 Analyze Watchlist")
        print("3. ➕ Add to Watchlist")
        print("4. ➖ Remove from Watchlist")
        print("5. 📋 Show Watchlist")
        print("6. 📈 Show Analysis History")
        print("7. 🔄 Real-time Monitoring")
        print("8. 💾 Save Analysis")
        print("9. 📖 Help")
        print("0. 🚪 Exit")
        print("="*40)
    
    def real_time_monitoring(self, symbol: str, interval: int = 60):
        """Real-time stock monitoring"""
        print(f"\n🔄 Real-time monitoring for {symbol.upper()}")
        print("Press Ctrl+C to stop monitoring")
        print("="*50)
        
        try:
            while True:
                analysis = self.analyze_stock(symbol, "1d")
                if analysis:
                    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - {symbol.upper()}: ${analysis.current_price:.2f} ({analysis.recommendation})")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")
    
    def save_analysis(self, filename: str = None):
        """Save analysis history to file"""
        if not filename:
            filename = f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            data = []
            for analysis in self.analysis_history:
                data.append({
                    'symbol': analysis.symbol,
                    'current_price': analysis.current_price,
                    'price_change': analysis.price_change,
                    'rsi': analysis.rsi,
                    'macd': analysis.macd,
                    'recommendation': analysis.recommendation,
                    'confidence': analysis.confidence,
                    'risk_level': analysis.risk_level,
                    'volatility': analysis.volatility,
                    'signals': analysis.signals,
                    'timestamp': analysis.timestamp.isoformat()
                })
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Analysis saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving analysis: {str(e)}")
    
    def run(self):
        """Main bot loop"""
        print("🚀 Starting Stock Market AI Bot...")
        print("🤖 AI-powered stock analysis and trading recommendations")
        
        while True:
            try:
                self.show_menu()
                choice = input("\n🎯 Enter your choice (0-9): ").strip()
                
                if choice == "1":
                    symbol = input("📊 Enter stock symbol: ").strip()
                    if symbol:
                        analysis = self.analyze_stock(symbol)
                        if analysis:
                            self.display_analysis(analysis)
                            self.analysis_history.append(analysis)
                
                elif choice == "2":
                    self.analyze_watchlist()
                
                elif choice == "3":
                    symbol = input("➕ Enter stock symbol to add: ").strip()
                    if symbol:
                        self.add_to_watchlist(symbol)
                
                elif choice == "4":
                    symbol = input("➖ Enter stock symbol to remove: ").strip()
                    if symbol:
                        self.remove_from_watchlist(symbol)
                
                elif choice == "5":
                    if self.watchlist:
                        print(f"\n📋 WATCHLIST ({len(self.watchlist)} stocks):")
                        for i, symbol in enumerate(self.watchlist, 1):
                            print(f"   {i}. {symbol}")
                    else:
                        print("📋 Watchlist is empty")
                
                elif choice == "6":
                    if self.analysis_history:
                        print(f"\n📈 ANALYSIS HISTORY ({len(self.analysis_history)} analyses):")
                        for i, analysis in enumerate(self.analysis_history[-10:], 1):
                            print(f"   {i}. {analysis.symbol}: ${analysis.current_price:.2f} ({analysis.recommendation}) - {analysis.timestamp.strftime('%H:%M:%S')}")
                    else:
                        print("📈 No analysis history")
                
                elif choice == "7":
                    symbol = input("🔄 Enter stock symbol for monitoring: ").strip()
                    if symbol:
                        interval = input("⏱️  Enter monitoring interval (seconds, default 60): ").strip()
                        interval = int(interval) if interval.isdigit() else 60
                        self.real_time_monitoring(symbol, interval)
                
                elif choice == "8":
                    filename = input("💾 Enter filename (or press Enter for default): ").strip()
                    self.save_analysis(filename if filename else None)
                
                elif choice == "9":
                    self.show_help()
                
                elif choice == "0":
                    print("👋 Thanks for using Stock Market AI Bot!")
                    break
                
                else:
                    print("❌ Invalid choice. Please try again.")
                
                input("\n⏸️  Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using Stock Market AI Bot!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                input("Press Enter to continue...")
    
    def show_help(self):
        """Show help information"""
        print("\n📖 STOCK MARKET AI BOT HELP")
        print("="*40)
        print("🤖 This AI bot analyzes stocks using:")
        print("   • Technical indicators (RSI, MACD, Moving Averages)")
        print("   • Price trends and momentum")
        print("   • Volume analysis")
        print("   • Volatility assessment")
        print("   • Risk evaluation")
        print("\n📊 Features:")
        print("   • Single stock analysis")
        print("   • Watchlist management")
        print("   • Real-time monitoring")
        print("   • Analysis history")
        print("   • Data export")
        print("\n💡 Tips:")
        print("   • Use popular symbols like AAPL, MSFT, GOOGL")
        print("   • Monitor stocks regularly for best results")
        print("   • Consider multiple timeframes for analysis")
        print("   • Always do your own research before trading")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stock Market AI Bot")
    parser.add_argument("--symbol", "-s", help="Analyze a specific stock symbol")
    parser.add_argument("--watchlist", "-w", action="store_true", help="Analyze watchlist")
    parser.add_argument("--monitor", "-m", help="Real-time monitoring for a symbol")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    bot = StockMarketBot()
    
    if args.symbol:
        # Quick analysis mode
        analysis = bot.analyze_stock(args.symbol)
        if analysis:
            bot.display_analysis(analysis)
    elif args.watchlist:
        # Watchlist analysis mode
        bot.analyze_watchlist()
    elif args.monitor:
        # Monitoring mode
        bot.real_time_monitoring(args.monitor, args.interval)
    else:
        # Interactive mode
        bot.run()

if __name__ == "__main__":
    main() 