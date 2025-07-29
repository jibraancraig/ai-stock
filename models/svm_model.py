#!/usr/bin/env python3
"""
SVM Model for Stock Market Prediction
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SVMModel:
    """Support Vector Machine model for stock price prediction"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', 
                 random_state: int = 42):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
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
        """Train the SVM model"""
        
        logger.info("Preparing data for SVM training...")
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(df, target_col, feature_cols)
        
        if tune_hyperparameters:
            logger.info("Tuning hyperparameters...")
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            self.model = SVR(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                random_state=self.random_state
            )
        
        logger.info("Training SVM model...")
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
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> SVR:
        """Tune hyperparameters using GridSearchCV"""
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        svm = SVR(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
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
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, f"{filepath}_model.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler"""
        self.model = joblib.load(f"{filepath}_model.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}") 