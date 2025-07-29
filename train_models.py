#!/usr/bin/env python3
"""
Model Training Pipeline
Trains all machine learning models for stock prediction
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
import logging
from typing import Dict, Any

from data_collector import StockDataCollector
from feature_engineering import FeatureEngineer
from models import LSTMModel, RandomForestModel, XGBoostModel, SVMModel, EnsembleModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains and manages all machine learning models"""
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        self.collector = StockDataCollector()
        self.engineer = FeatureEngineer()
        os.makedirs(models_dir, exist_ok=True)
        
    def train_all_models(self, symbol: str, period: str = "2y", 
                        save_models: bool = True) -> Dict[str, Any]:
        """Train all models for a given stock symbol"""
        
        logger.info(f"Starting training pipeline for {symbol}")
        
        # Step 1: Collect data
        logger.info("Step 1: Collecting stock data...")
        data = self.collector.fetch_stock_data(symbol, period=period)
        
        # Step 2: Engineer features
        logger.info("Step 2: Engineering features...")
        features_df = self.engineer.engineer_all_features(data)
        
        # Step 3: Train models
        logger.info("Step 3: Training models...")
        results = {}
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestModel(n_estimators=100, max_depth=20)
        rf_metrics = rf_model.train(features_df, tune_hyperparameters=False)
        results['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics
        }
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = XGBoostModel(n_estimators=100, max_depth=6)
        xgb_metrics = xgb_model.train(features_df, tune_hyperparameters=False)
        results['xgboost'] = {
            'model': xgb_model,
            'metrics': xgb_metrics
        }
        
        # Train SVM
        logger.info("Training SVM...")
        svm_model = SVMModel(kernel='rbf', C=10)
        svm_metrics = svm_model.train(features_df, tune_hyperparameters=False)
        results['svm'] = {
            'model': svm_model,
            'metrics': svm_metrics
        }
        
        # Train Ensemble
        logger.info("Training Ensemble Model...")
        ensemble_model = EnsembleModel()
        ensemble_metrics = ensemble_model.train(features_df)
        results['ensemble'] = {
            'model': ensemble_model,
            'metrics': ensemble_metrics
        }
        
        # Train LSTM (optional - takes longer)
        try:
            logger.info("Training LSTM (this may take a while)...")
            lstm_model = LSTMModel(sequence_length=60, prediction_days=1)
            lstm_metrics = lstm_model.train(features_df, epochs=50)  # Reduced epochs for speed
            results['lstm'] = {
                'model': lstm_model,
                'metrics': lstm_metrics
            }
        except Exception as e:
            logger.warning(f"LSTM training failed: {str(e)}")
            results['lstm'] = {'error': str(e)}
        
        # Step 4: Save models
        if save_models:
            logger.info("Step 4: Saving models...")
            self._save_models(symbol, results)
        
        # Step 5: Generate summary
        logger.info("Step 5: Generating training summary...")
        summary = self._generate_summary(symbol, results, features_df)
        
        return results, summary
    
    def _save_models(self, symbol: str, results: Dict[str, Any]):
        """Save all trained models"""
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
                
            model = result['model']
            filepath = os.path.join(self.models_dir, f"{symbol}_{model_name}")
            
            try:
                model.save_model(filepath)
                logger.info(f"Saved {model_name} model")
            except Exception as e:
                logger.error(f"Failed to save {model_name} model: {str(e)}")
    
    def _generate_summary(self, symbol: str, results: Dict[str, Any], 
                         features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate training summary"""
        
        summary = {
            'symbol': symbol,
            'training_date': datetime.now().isoformat(),
            'data_info': {
                'total_records': len(features_df),
                'date_range': f"{features_df['date'].min()} to {features_df['date'].max()}",
                'features_count': len(self.engineer.feature_columns)
            },
            'model_performance': {},
            'best_model': None,
            'recommendations': []
        }
        
        best_r2 = -float('inf')
        best_model = None
        
        for model_name, result in results.items():
            if 'error' in result:
                summary['model_performance'][model_name] = {'error': result['error']}
                continue
            
            metrics = result['metrics']
            summary['model_performance'][model_name] = {
                'test_rmse': metrics.get('test_rmse', 0),
                'test_mae': metrics.get('test_mae', 0),
                'test_r2': metrics.get('test_r2', 0)
            }
            
            # Track best model
            if metrics.get('test_r2', 0) > best_r2:
                best_r2 = metrics.get('test_r2', 0)
                best_model = model_name
        
        summary['best_model'] = best_model
        
        # Generate recommendations
        if best_model:
            summary['recommendations'].append(f"Best performing model: {best_model} (R²: {best_r2:.4f})")
        
        # Add feature importance if available
        if 'random_forest' in results and 'error' not in results['random_forest']:
            rf_model = results['random_forest']['model']
            top_features = rf_model.get_feature_importance(top_n=10)
            summary['top_features'] = top_features.to_dict('records')
        
        return summary
    
    def load_trained_models(self, symbol: str) -> Dict[str, Any]:
        """Load all trained models for a symbol"""
        
        models = {}
        
        model_types = ['random_forest', 'xgboost', 'svm', 'ensemble', 'lstm']
        
        for model_type in model_types:
            filepath = os.path.join(self.models_dir, f"{symbol}_{model_type}")
            
            try:
                if model_type == 'lstm':
                    model = LSTMModel()
                elif model_type == 'random_forest':
                    model = RandomForestModel()
                elif model_type == 'xgboost':
                    model = XGBoostModel()
                elif model_type == 'svm':
                    model = SVMModel()
                elif model_type == 'ensemble':
                    model = EnsembleModel()
                
                model.load_model(filepath)
                models[model_type] = model
                logger.info(f"Loaded {model_type} model")
                
            except Exception as e:
                logger.warning(f"Failed to load {model_type} model: {str(e)}")
        
        return models

def main():
    parser = argparse.ArgumentParser(description='Train machine learning models for stock prediction')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='2y', help='Data period')
    parser.add_argument('--output', type=str, default='training_summary.json', help='Output summary file')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        # Train all models
        results, summary = trainer.train_all_models(args.symbol, args.period)
        
        # Save summary
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"TRAINING SUMMARY FOR {args.symbol}")
        print(f"{'='*50}")
        print(f"Training Date: {summary['training_date']}")
        print(f"Data Records: {summary['data_info']['total_records']}")
        print(f"Features: {summary['data_info']['features_count']}")
        print(f"Date Range: {summary['data_info']['date_range']}")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"{'Model':<15} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10}")
        print(f"{'-'*50}")
        
        for model_name, perf in summary['model_performance'].items():
            if 'error' in perf:
                print(f"{model_name:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10}")
            else:
                print(f"{model_name:<15} {perf['test_rmse']:<12.4f} {perf['test_mae']:<12.4f} {perf['test_r2']:<10.4f}")
        
        if summary['best_model']:
            print(f"\nBest Model: {summary['best_model']}")
        
        if 'top_features' in summary:
            print(f"\nTop 5 Most Important Features:")
            for i, feature in enumerate(summary['top_features'][:5]):
                print(f"{i+1}. {feature['feature']} (importance: {feature['importance']:.4f})")
        
        print(f"\nSummary saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 