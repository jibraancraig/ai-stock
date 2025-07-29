#!/usr/bin/env python3
"""
Ensemble Model for Stock Market Prediction
Combines multiple models for improved accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import List, Optional, Dict, Any
import logging

from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .svm_model import SVMModel

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model combining multiple algorithms"""
    
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights or [0.3, 0.3, 0.2, 0.2]  # RF, XGB, SVM, Linear
        self.models = {}
        self.scaler = StandardScaler()
        self.ensemble = None
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                    feature_cols: Optional[List[str]] = None, test_size: float = 0.2) -> tuple:
        """Prepare data for training"""
        
        if feature_cols is None:
            exclude_cols = ['date', 'Symbol', target_col]
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train(self, df: pd.DataFrame, target_col: str = 'Close', 
              feature_cols: Optional[List[str]] = None) -> dict:
        """Train the ensemble model"""
        
        logger.info("Preparing data for ensemble training...")
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(df, target_col, feature_cols)
        
        # Create individual models
        self.models['rf'] = RandomForestModel(n_estimators=100, max_depth=20)
        self.models['xgb'] = XGBoostModel(n_estimators=100, max_depth=6)
        self.models['svm'] = SVMModel(kernel='rbf', C=10)
        
        # Train individual models
        logger.info("Training individual models...")
        
        # Random Forest
        rf_metrics = self.models['rf'].train(df, target_col, feature_cols)
        logger.info(f"Random Forest - Test R²: {rf_metrics['test_r2']:.4f}")
        
        # XGBoost
        xgb_metrics = self.models['xgb'].train(df, target_col, feature_cols)
        logger.info(f"XGBoost - Test R²: {xgb_metrics['test_r2']:.4f}")
        
        # SVM
        svm_metrics = self.models['svm'].train(df, target_col, feature_cols)
        logger.info(f"SVM - Test R²: {svm_metrics['test_r2']:.4f}")
        
        # Create ensemble
        estimators = [
            ('rf', self.models['rf'].model),
            ('xgb', self.models['xgb'].model),
            ('svm', self.models['svm'].model),
            ('linear', LinearRegression())
        ]
        
        self.ensemble = VotingRegressor(
            estimators=estimators,
            weights=self.weights
        )
        
        # Train ensemble
        logger.info("Training ensemble model...")
        self.ensemble.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.ensemble.predict(X_train)
        y_test_pred = self.ensemble.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'individual_models': {
                'random_forest': rf_metrics,
                'xgboost': xgb_metrics,
                'svm': svm_metrics
            }
        }
        
        self.is_trained = True
        logger.info(f"Ensemble training completed. Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Make predictions using the ensemble model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if feature_cols is None:
            exclude_cols = ['date', 'Symbol', 'Close']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        X_scaled = self.scaler.transform(X)
        
        return self.ensemble.predict(X_scaled)
    
    def get_individual_predictions(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(df, feature_cols)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the ensemble model and individual models"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save ensemble
        joblib.dump(self.ensemble, f"{filepath}_ensemble.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save individual models
        for name, model in self.models.items():
            model.save_model(f"{filepath}_{name}")
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the ensemble model and individual models"""
        
        # Load ensemble
        self.ensemble = joblib.load(f"{filepath}_ensemble.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        # Load individual models
        self.models['rf'] = RandomForestModel()
        self.models['rf'].load_model(f"{filepath}_rf")
        
        self.models['xgb'] = XGBoostModel()
        self.models['xgb'].load_model(f"{filepath}_xgb")
        
        self.models['svm'] = SVMModel()
        self.models['svm'].load_model(f"{filepath}_svm")
        
        self.is_trained = True
        logger.info(f"Ensemble model loaded from {filepath}") 