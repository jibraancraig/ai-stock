#!/usr/bin/env python3
"""
LSTM Neural Network Model for Stock Market Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM Neural Network for stock price prediction"""
    
    def __init__(self, sequence_length: int = 60, prediction_days: int = 1):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False
        
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i:i+self.prediction_days])
            
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                    feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        
        if feature_cols is None:
            # Use all numeric columns except date and target
            exclude_cols = ['date', 'Symbol', target_col]
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Prepare features
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, target_scaled)
        
        # Split into train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture"""
        
        model = keras.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(100, return_sequences=True, input_shape=input_shape, dropout=0.2),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(100, return_sequences=True, dropout=0.2),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(100, return_sequences=False, dropout=0.2),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.prediction_days, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, target_col: str = 'Close', 
              feature_cols: Optional[List[str]] = None, epochs: int = 100, 
              batch_size: int = 32, validation_split: float = 0.2) -> dict:
        """Train the LSTM model"""
        
        logger.info("Preparing data for LSTM training...")
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col, feature_cols)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Reduce learning rate callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train model
        logger.info("Training LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        train_loss, train_mae = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_train_pred = self.scaler.inverse_transform(y_train_pred)
        y_test_pred = self.scaler.inverse_transform(y_test_pred)
        y_train_actual = self.scaler.inverse_transform(y_train)
        y_test_actual = self.scaler.inverse_transform(y_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
        train_r2 = r2_score(y_train_actual, y_train_pred)
        test_r2 = r2_score(y_test_actual, y_test_pred)
        
        metrics = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        self.is_trained = True
        
        logger.info(f"Training completed. Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Make predictions using the trained model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if feature_cols is None:
            exclude_cols = ['date', 'Symbol', 'Close']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Prepare features
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)
        
        # Create sequences for prediction
        X_pred = []
        for i in range(self.sequence_length, len(features_scaled)):
            X_pred.append(features_scaled[i-self.sequence_length:i])
        
        X_pred = np.array(X_pred)
        
        # Make predictions
        predictions_scaled = self.model.predict(X_pred)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions
    
    def predict_next_days(self, df: pd.DataFrame, days: int = 5, 
                         feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Predict next N days"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if feature_cols is None:
            exclude_cols = ['date', 'Symbol', 'Close']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Get the last sequence
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)
        last_sequence = features_scaled[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, len(feature_cols))
            
            # Make prediction
            pred_scaled = self.model.predict(X, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred)
            
            # Update sequence for next prediction (simple approach)
            # In a more sophisticated approach, you'd need to update all features
            current_sequence = np.roll(current_sequence, -1, axis=0)
            # This is a simplified approach - in practice you'd need to update all features
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Save the trained model and scalers"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save scalers
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.feature_scaler, f"{filepath}_feature_scaler.pkl")
        
        # Save model info
        model_info = {
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'is_trained': self.is_trained
        }
        joblib.dump(model_info, f"{filepath}_info.pkl")
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scalers"""
        
        # Load model
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        
        # Load scalers
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.feature_scaler = joblib.load(f"{filepath}_feature_scaler.pkl")
        
        # Load model info
        model_info = joblib.load(f"{filepath}_info.pkl")
        self.sequence_length = model_info['sequence_length']
        self.prediction_days = model_info['prediction_days']
        self.is_trained = model_info['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list) 