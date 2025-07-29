#!/usr/bin/env python3
"""
Quick Start Demo
Demonstrates the complete Stock Market AI workflow
"""

import sys
import os
from datetime import datetime

def main():
    print("🚀 Stock Market AI - Quick Start Demo")
    print("=" * 50)
    
    # Check if we have the required modules
    try:
        from data_collector import StockDataCollector
        from feature_engineering import FeatureEngineer
        from investment_advisor import InvestmentAdvisor
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return 1
    
    # Demo parameters
    symbol = "AAPL"
    investment_amount = 10000
    
    print(f"\n📊 Analyzing {symbol} with ${investment_amount:,} investment")
    print("-" * 50)
    
    try:
        # Step 1: Data Collection
        print("\n1️⃣ Collecting stock data...")
        collector = StockDataCollector()
        data = collector.fetch_stock_data(symbol, period="1y")
        print(f"   ✅ Collected {len(data)} data points")
        
        # Step 2: Feature Engineering
        print("\n2️⃣ Engineering features...")
        engineer = FeatureEngineer()
        features_df = engineer.engineer_all_features(data)
        print(f"   ✅ Created {len(engineer.feature_columns)} features")
        
        # Step 3: Investment Recommendation
        print("\n3️⃣ Generating investment recommendation...")
        advisor = InvestmentAdvisor()
        recommendation = advisor.get_recommendation(symbol, investment_amount)
        
        if 'error' in recommendation:
            print(f"   ❌ Error: {recommendation['error']}")
            print("   💡 Try training models first: python train_models.py --symbol AAPL")
            return 1
        
        # Display results
        print(f"\n📈 ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Symbol: {recommendation['symbol']}")
        print(f"Current Price: ${recommendation['current_price']:.2f}")
        print(f"Signal: {recommendation['signal']}")
        print(f"Confidence: {recommendation['confidence']:.1%}")
        print(f"Price Target: ${recommendation['price_target']:.2f}")
        print(f"Stop Loss: ${recommendation['stop_loss']:.2f}")
        print(f"Risk Level: {recommendation['risk_level']}")
        
        print(f"\n💰 POSITION SIZING")
        print("-" * 30)
        pos = recommendation['position_sizing']
        print(f"Recommended Shares: {pos['recommended_shares']}")
        print(f"Position Value: ${pos['position_value']:.2f}")
        print(f"Portfolio Allocation: {pos['position_percentage']:.1f}%")
        
        print(f"\n🎯 REASONING")
        print("-" * 30)
        for i, reason in enumerate(recommendation['reasoning'][:5], 1):
            print(f"{i}. {reason}")
        
        print(f"\n⚠️  RISK ASSESSMENT")
        print("-" * 30)
        risk = recommendation['risk_assessment']
        print(f"Volatility Risk: {risk['volatility_risk']}")
        print(f"Market Risk: {risk['market_risk']}")
        print(f"Prediction Risk: {risk['prediction_risk']}")
        print(f"Overall Risk: {risk['overall_risk']}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   • Train models: python train_models.py --symbol {symbol}")
        print(f"   • Get predictions: python predict.py --symbol {symbol}")
        print(f"   • Launch dashboard: python app.py")
        print(f"   • Get recommendations: python investment_advisor.py --symbol {symbol}")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 