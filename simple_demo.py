#!/usr/bin/env python3
"""
Simple Stock Market AI Demo
Works with basic packages (no XGBoost required)
"""

import sys
import os
from datetime import datetime

def main():
    print("🚀 Stock Market AI - Simple Demo")
    print("=" * 50)
    
    # Check if we have the required modules
    try:
        from data_collector import StockDataCollector
        from feature_engineering import FeatureEngineer
        print("✅ Core modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    
    # Demo parameters
    symbol = "AAPL"
    
    print(f"\n📊 Analyzing {symbol}")
    print("-" * 50)
    
    try:
        # Step 1: Data Collection
        print("\n1️⃣ Collecting stock data...")
        collector = StockDataCollector()
        data = collector.fetch_stock_data(symbol, period="6mo")
        print(f"   ✅ Collected {len(data)} data points")
        print(f"   📅 Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"   💰 Latest price: ${data['Close'].iloc[-1]:.2f}")
        
        # Step 2: Feature Engineering
        print("\n2️⃣ Engineering features...")
        engineer = FeatureEngineer()
        features_df = engineer.engineer_all_features(data)
        print(f"   ✅ Created {len(engineer.feature_columns)} features")
        
        # Step 3: Show some key features
        print("\n3️⃣ Key Technical Indicators:")
        latest = features_df.iloc[-1]
        
        indicators = [
            ('RSI', latest.get('rsi', 'N/A')),
            ('MACD', latest.get('macd', 'N/A')),
            ('Bollinger Band Position', latest.get('bb_position', 'N/A')),
            ('Price vs MA20', f"{latest.get('price_vs_ma20', 0)*100:.2f}%"),
            ('Volume Ratio', latest.get('volume_ratio', 'N/A')),
            ('Volatility (20-day)', f"{latest.get('volatility_20', 0)*100:.2f}%")
        ]
        
        for name, value in indicators:
            if value != 'N/A':
                print(f"   📈 {name}: {value}")
        
        # Step 4: Basic Analysis
        print("\n4️⃣ Basic Analysis:")
        
        # Price trend
        current_price = data['Close'].iloc[-1]
        price_1m_ago = data['Close'].iloc[-20] if len(data) >= 20 else data['Close'].iloc[0]
        price_change = ((current_price - price_1m_ago) / price_1m_ago) * 100
        
        print(f"   💰 Current Price: ${current_price:.2f}")
        print(f"   📈 1-Month Change: {price_change:+.2f}%")
        
        # Volume analysis
        avg_volume = data['Volume'].mean()
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        print(f"   📊 Average Volume: {avg_volume:,.0f}")
        print(f"   📊 Current Volume: {current_volume:,.0f} ({volume_ratio:.1f}x average)")
        
        # Simple recommendation
        print("\n5️⃣ Simple Recommendation:")
        
        if price_change > 5:
            trend = "Strong Uptrend"
            sentiment = "Bullish"
        elif price_change > 0:
            trend = "Uptrend"
            sentiment = "Slightly Bullish"
        elif price_change > -5:
            trend = "Sideways"
            sentiment = "Neutral"
        else:
            trend = "Downtrend"
            sentiment = "Bearish"
        
        print(f"   🎯 Trend: {trend}")
        print(f"   📊 Sentiment: {sentiment}")
        
        # Risk assessment
        volatility = latest.get('volatility_20', 0)
        if volatility > 0.3:
            risk_level = "High"
        elif volatility > 0.2:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        print(f"   ⚠️  Risk Level: {risk_level}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   • Install additional packages for full ML capabilities")
        print(f"   • Train models: python3 train_models.py --symbol {symbol}")
        print(f"   • Launch dashboard: python3 app.py")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 