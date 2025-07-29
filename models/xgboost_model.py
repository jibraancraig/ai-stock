#!/usr/bin/env python3
"""
XGBoost Model for Stock Market Prediction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost model for stock price prediction"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
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
            X_scaled, y, test_size=test_size, random_state=self.random_state, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train(self, df: pd.DataFrame, target_col: str = 'Close', 
              feature_cols: Optional[List[str]] = None, tune_hyperparameters: bool = False) -> dict:
        """Train the XGBoost model"""
        
        logger.info("Preparing data for XGBoost training...")
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(df, target_col, feature_cols)
        
        if tune_hyperparameters:
            logger.info("Tuning hyperparameters...")
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        logger.info("Training XGBoost model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        self.is_trained = True
        logger.info(f"Training completed. Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        return metrics
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """Tune hyperparameters using GridSearchCV"""
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Make predictions using the trained model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if feature_cols is None:
            exclude_cols = ['date', 'Symbol', 'Close']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, f"{filepath}_model.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.feature_importance, f"{filepath}_importance.pkl")
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler"""
        self.model = joblib.load(f"{filepath}_model.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.feature_importance = joblib.load(f"{filepath}_importance.pkl")
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}") 