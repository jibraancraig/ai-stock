#!/usr/bin/env python3
"""
Quick Stock Prediction Script
Get predictions for a stock symbol
"""

import argparse
import sys
from data_collector import StockDataCollector
from feature_engineering import FeatureEngineer
from train_models import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Get stock price predictions')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        collector = StockDataCollector()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        
        print(f"Getting predictions for {args.symbol}...")
        
        # Get latest data
        data = collector.fetch_stock_data(args.symbol, period="6mo")
        features_df = engineer.engineer_all_features(data)
        
        # Load trained models
        models = trainer.load_trained_models(args.symbol)
        
        if not models:
            print(f"No trained models found for {args.symbol}")
            print("Please train models first using: python train_models.py --symbol {args.symbol}")
            return 1
        
        current_price = features_df['Close'].iloc[-1]
        print(f"\nCurrent Price: ${current_price:.2f}")
        print(f"Predicting next {args.days} days...")
        
        # Get predictions from each model
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_next_days'):
                    predictions = model.predict_next_days(features_df, days=args.days)
                    print(f"\n{model_name.upper()} Predictions:")
                    for i, pred in enumerate(predictions, 1):
                        change = ((pred - current_price) / current_price) * 100
                        print(f"  Day {i}: ${pred:.2f} ({change:+.2f}%)")
                else:
                    pred = model.predict(features_df)
                    if len(pred) > 0:
                        latest_pred = pred[-1]
                        change = ((latest_pred - current_price) / current_price) * 100
                        print(f"\n{model_name.upper()} Prediction:")
                        print(f"  Next: ${latest_pred:.2f} ({change:+.2f}%)")
                        
            except Exception as e:
                print(f"\n{model_name.upper()}: Error - {str(e)}")
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 