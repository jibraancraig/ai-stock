#!/usr/bin/env python3
"""
Stock Market Data Collector
Fetches historical and real-time stock data for analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os
import json
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataCollector:
    """Collects and manages stock market data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol} over {period}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Add date index as column
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'date'}, inplace=True)
            
            # Ensure all required columns exist
            required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Add additional features
            data = self._add_basic_features(data)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic calculated features to the dataset"""
        
        # Price changes
        data['price_change'] = data['Close'].diff()
        data['price_change_pct'] = data['Close'].pct_change()
        
        # Volume features
        data['volume_ma'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma']
        
        # Price ranges
        data['daily_range'] = data['High'] - data['Low']
        data['daily_range_pct'] = data['daily_range'] / data['Close']
        
        # Moving averages
        data['ma_5'] = data['Close'].rolling(window=5).mean()
        data['ma_20'] = data['Close'].rolling(window=20).mean()
        data['ma_50'] = data['Close'].rolling(window=50).mean()
        
        # Price position relative to moving averages
        data['price_vs_ma5'] = data['Close'] / data['ma_5'] - 1
        data['price_vs_ma20'] = data['Close'] / data['ma_20'] - 1
        data['price_vs_ma50'] = data['Close'] / data['ma_50'] - 1
        
        return data
    
    def save_data(self, data: pd.DataFrame, symbol: str) -> str:
        """Save data to CSV file"""
        filename = f"{self.data_dir}/{symbol}_data.csv"
        data.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
        return filename
    
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        filename = f"{self.data_dir}/{symbol}_data.csv"
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            data['date'] = pd.to_datetime(data['date'])
            logger.info(f"Data loaded from {filename}")
            return data
        else:
            logger.warning(f"Data file not found: {filename}")
            return None
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Fetch stock market data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='2y', help='Time period')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--save', action='store_true', help='Save data to file')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = StockDataCollector()
    
    try:
        # Fetch data
        data = collector.fetch_stock_data(args.symbol, args.period, args.interval)
        
        # Get stock info
        info = collector.get_stock_info(args.symbol)
        
        print(f"\nStock Information:")
        print(f"Symbol: {info['symbol']}")
        print(f"Name: {info['name']}")
        print(f"Sector: {info['sector']}")
        print(f"Industry: {info['industry']}")
        
        print(f"\nData Summary:")
        print(f"Records: {len(data)}")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"Latest close: ${data['Close'].iloc[-1]:.2f}")
        print(f"Price change: {data['price_change_pct'].iloc[-1]*100:.2f}%")
        
        if args.save:
            filename = collector.save_data(data, args.symbol)
            print(f"\nData saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 