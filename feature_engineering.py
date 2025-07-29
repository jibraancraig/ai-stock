#!/usr/bin/env python3
"""
Feature Engineering for Stock Market Analysis
Calculates technical indicators and mathematical features
"""

import pandas as pd
import numpy as np
import ta
from scipy import stats
from scipy.signal import savgol_filter
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineers features for stock market prediction"""
    
    def __init__(self):
        self.feature_columns = []
    
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering techniques"""
        logger.info("Starting feature engineering...")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Mathematical features
        df = self._add_mathematical_features(df)
        
        # Calculus-based features
        df = self._add_calculus_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Pattern recognition features
        df = self._add_pattern_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Clean up infinite and NaN values
        df = self._clean_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_columns)}")
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        
        # Trend indicators
        df['sma_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df['macd'] = ta.trend.macd(df['Close'])
        df['macd_signal'] = ta.trend.macd_signal(df['Close'])
        df['macd_diff'] = ta.trend.macd_diff(df['Close'])
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['Close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['Close'])
        
        # Stochastic
        df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # Commodity Channel Index
        df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # Average True Range
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Parabolic SAR
        df['psar'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # On Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Accumulation/Distribution Line
        df['adl'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        self.feature_columns.extend([
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_lower', 
            'bb_middle', 'bb_width', 'bb_position', 'rsi', 'stoch_k', 
            'stoch_d', 'williams_r', 'cci', 'atr', 'psar', 'adx', 
            'mfi', 'obv', 'adl', 'cmf'
        ])
        
        return df
    
    def _add_mathematical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mathematical features"""
        
        # Price ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['open_close_ratio'] = df['Open'] / df['Close']
        
        # Log returns (for better statistical properties)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price momentum
        for period in [1, 3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'log_momentum_{period}'] = np.log(df['Close'] / df['Close'].shift(period))
        
        # Volume momentum
        for period in [1, 3, 5, 10]:
            df[f'volume_momentum_{period}'] = df['Volume'] / df['Volume'].shift(period) - 1
        
        # Price acceleration (second derivative)
        df['price_acceleration'] = df['price_change'].diff()
        
        # Volume acceleration
        df['volume_acceleration'] = df['Volume'].diff().diff()
        
        # Price velocity (rate of change)
        df['price_velocity'] = df['Close'].diff() / df['Close'].shift(1)
        
        # Volume velocity
        df['volume_velocity'] = df['Volume'].diff() / df['Volume'].shift(1)
        
        self.feature_columns.extend([
            'high_low_ratio', 'open_close_ratio', 'log_return',
            'momentum_1', 'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
            'log_momentum_1', 'log_momentum_3', 'log_momentum_5', 'log_momentum_10', 'log_momentum_20',
            'volume_momentum_1', 'volume_momentum_3', 'volume_momentum_5', 'volume_momentum_10',
            'price_acceleration', 'volume_acceleration', 'price_velocity', 'volume_velocity'
        ])
        
        return df
    
    def _add_calculus_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculus-based features"""
        
        # First derivative (rate of change)
        df['price_derivative_1'] = df['Close'].diff()
        df['volume_derivative_1'] = df['Volume'].diff()
        
        # Second derivative (acceleration)
        df['price_derivative_2'] = df['price_derivative_1'].diff()
        df['volume_derivative_2'] = df['volume_derivative_1'].diff()
        
        # Third derivative (jerk)
        df['price_derivative_3'] = df['price_derivative_2'].diff()
        df['volume_derivative_3'] = df['volume_derivative_2'].diff()
        
        # Smooth derivatives using Savitzky-Golay filter
        try:
            window_length = min(15, len(df) // 4)
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                df['price_smooth_derivative'] = savgol_filter(df['Close'], window_length, 2, deriv=1)
                df['volume_smooth_derivative'] = savgol_filter(df['Volume'], window_length, 2, deriv=1)
        except:
            df['price_smooth_derivative'] = df['price_derivative_1']
            df['volume_smooth_derivative'] = df['volume_derivative_1']
        
        # Integral features (cumulative sums)
        df['price_integral_5'] = df['Close'].rolling(5).sum()
        df['volume_integral_5'] = df['Volume'].rolling(5).sum()
        
        # Rate of change ratios
        df['price_volume_ratio'] = df['price_derivative_1'] / (df['volume_derivative_1'] + 1e-8)
        
        self.feature_columns.extend([
            'price_derivative_1', 'volume_derivative_1', 'price_derivative_2', 
            'volume_derivative_2', 'price_derivative_3', 'volume_derivative_3',
            'price_smooth_derivative', 'volume_smooth_derivative',
            'price_integral_5', 'volume_integral_5', 'price_volume_ratio'
        ])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'price_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'price_std_{window}'] = df['Close'].rolling(window).std()
            df[f'price_skew_{window}'] = df['Close'].rolling(window).skew()
            df[f'price_kurt_{window}'] = df['Close'].rolling(window).kurt()
            df[f'volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['Volume'].rolling(window).std()
        
        # Z-scores
        for window in [10, 20]:
            df[f'price_zscore_{window}'] = (df['Close'] - df[f'price_mean_{window}']) / (df[f'price_std_{window}'] + 1e-8)
            df[f'volume_zscore_{window}'] = (df['Volume'] - df[f'volume_mean_{window}']) / (df[f'volume_std_{window}'] + 1e-8)
        
        # Percentile ranks
        for window in [10, 20]:
            df[f'price_percentile_{window}'] = df['Close'].rolling(window).rank(pct=True)
            df[f'volume_percentile_{window}'] = df['Volume'].rolling(window).rank(pct=True)
        
        # Correlation features
        df['price_volume_corr_10'] = df['Close'].rolling(10).corr(df['Volume'])
        df['price_volume_corr_20'] = df['Close'].rolling(20).corr(df['Volume'])
        
        self.feature_columns.extend([
            'price_mean_5', 'price_mean_10', 'price_mean_20', 'price_std_5', 'price_std_10', 'price_std_20',
            'price_skew_5', 'price_skew_10', 'price_skew_20', 'price_kurt_5', 'price_kurt_10', 'price_kurt_20',
            'volume_mean_5', 'volume_mean_10', 'volume_mean_20', 'volume_std_5', 'volume_std_10', 'volume_std_20',
            'price_zscore_10', 'price_zscore_20', 'volume_zscore_10', 'volume_zscore_20',
            'price_percentile_10', 'price_percentile_20', 'volume_percentile_10', 'volume_percentile_20',
            'price_volume_corr_10', 'price_volume_corr_20'
        ])
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        
        # Candlestick patterns
        df['doji'] = ((df['High'] - df['Low']) <= 0.1 * (df['Open'] - df['Close']).abs()).astype(int)
        df['hammer'] = ((df['Low'] - df['Open']) > 2 * (df['Open'] - df['Close'])) & (df['Close'] > df['Open']).astype(int)
        df['shooting_star'] = ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])) & (df['Close'] < df['Open']).astype(int)
        
        # Gap patterns
        df['gap_up'] = (df['Open'] > df['High'].shift(1)).astype(int)
        df['gap_down'] = (df['Open'] < df['Low'].shift(1)).astype(int)
        
        # Price patterns
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['higher_close'] = (df['Close'] > df['Close'].shift(1)).astype(int)
        
        # Volume patterns
        df['high_volume'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
        df['low_volume'] = (df['Volume'] < df['Volume'].rolling(20).mean() * 0.5).astype(int)
        
        # Trend patterns
        df['uptrend_5'] = (df['Close'] > df['Close'].shift(5)).astype(int)
        df['downtrend_5'] = (df['Close'] < df['Close'].shift(5)).astype(int)
        df['uptrend_10'] = (df['Close'] > df['Close'].shift(10)).astype(int)
        df['downtrend_10'] = (df['Close'] < df['Close'].shift(10)).astype(int)
        
        self.feature_columns.extend([
            'doji', 'hammer', 'shooting_star', 'gap_up', 'gap_down',
            'higher_high', 'lower_low', 'higher_close', 'high_volume', 'low_volume',
            'uptrend_5', 'downtrend_5', 'uptrend_10', 'downtrend_10'
        ])
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        
        # Historical volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (uses high-low range)
        for window in [5, 10, 20]:
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(df['High'] / df['Low']) ** 2).rolling(window).mean()) * 252
            )
        
        # Garman-Klass volatility
        for window in [5, 10, 20]:
            df[f'garman_klass_vol_{window}'] = np.sqrt(
                (0.5 * (np.log(df['High'] / df['Low']) ** 2) - 
                 (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)).rolling(window).mean() * 252
            )
        
        # Volatility ratios
        df['volatility_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        df['parkinson_ratio_5_20'] = df['parkinson_vol_5'] / (df['parkinson_vol_20'] + 1e-8)
        
        self.feature_columns.extend([
            'volatility_5', 'volatility_10', 'volatility_20',
            'parkinson_vol_5', 'parkinson_vol_10', 'parkinson_vol_20',
            'garman_klass_vol_5', 'garman_klass_vol_10', 'garman_klass_vol_20',
            'volatility_ratio_5_20', 'parkinson_ratio_5_20'
        ])
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        
        # Rate of change
        for period in [1, 3, 5, 10, 20]:
            df[f'roc_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Momentum oscillator
        for period in [10, 20]:
            df[f'momentum_osc_{period}'] = df['Close'] - df['Close'].shift(period)
        
        # Rate of change momentum
        for period in [5, 10]:
            df[f'roc_momentum_{period}'] = df[f'roc_{period}'] - df[f'roc_{period}'].shift(period)
        
        # Price momentum divergence
        df['price_momentum_divergence'] = df['momentum_10'] - df['momentum_20']
        
        # Volume momentum divergence
        df['volume_momentum_divergence'] = df['volume_momentum_5'] - df['volume_momentum_10']
        
        self.feature_columns.extend([
            'roc_1', 'roc_3', 'roc_5', 'roc_10', 'roc_20',
            'momentum_osc_10', 'momentum_osc_20',
            'roc_momentum_5', 'roc_momentum_10',
            'price_momentum_divergence', 'volume_momentum_divergence'
        ])
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean infinite and NaN values"""
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values for most features
        feature_cols = [col for col in df.columns if col in self.feature_columns]
        df[feature_cols] = df[feature_cols].fillna(method='ffill')
        
        # Fill remaining NaN values with 0
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame, target: str = 'price_change_pct') -> pd.DataFrame:
        """Rank features by correlation with target"""
        
        feature_cols = [col for col in df.columns if col in self.feature_columns]
        
        correlations = []
        for col in feature_cols:
            corr = df[col].corr(df[target])
            correlations.append({'feature': col, 'correlation': corr})
        
        ranking = pd.DataFrame(correlations)
        ranking = ranking.sort_values('correlation', key=abs, ascending=False)
        
        return ranking

def main():
    """Test the feature engineering"""
    from data_collector import StockDataCollector
    
    # Fetch sample data
    collector = StockDataCollector()
    data = collector.fetch_stock_data('AAPL', period='1y')
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_all_features(data)
    
    print(f"Original columns: {len(data.columns)}")
    print(f"Features added: {len(engineer.feature_columns)}")
    print(f"Total columns: {len(features_df.columns)}")
    
    # Show feature importance
    ranking = engineer.get_feature_importance_ranking(features_df)
    print("\nTop 10 most correlated features:")
    print(ranking.head(10))

if __name__ == "__main__":
    main() 