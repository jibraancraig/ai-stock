#!/usr/bin/env python3
"""
Investment Advisor
Provides buy/sell recommendations based on AI predictions and technical analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from data_collector import StockDataCollector
from feature_engineering import FeatureEngineer
from train_models import ModelTrainer

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    signal: SignalType
    confidence: float
    price_target: float
    stop_loss: float
    reasoning: List[str]
    risk_level: str
    time_horizon: str

class InvestmentAdvisor:
    """Provides investment recommendations based on AI predictions"""
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        self.collector = StockDataCollector()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer(models_dir)
        
    def get_recommendation(self, symbol: str, investment_amount: float = 10000,
                          risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """Get comprehensive investment recommendation"""
        
        logger.info(f"Generating investment recommendation for {symbol}")
        
        try:
            # Get latest data
            data = self.collector.fetch_stock_data(symbol, period="6mo")
            features_df = self.engineer.engineer_all_features(data)
            
            # Load trained models
            models = self.trainer.load_trained_models(symbol)
            
            if not models:
                raise ValueError(f"No trained models found for {symbol}")
            
            # Get predictions from all models
            predictions = self._get_model_predictions(features_df, models)
            
            # Analyze technical indicators
            technical_analysis = self._analyze_technical_indicators(features_df)
            
            # Generate trading signal
            signal = self._generate_trading_signal(
                predictions, technical_analysis, features_df, 
                investment_amount, risk_tolerance
            )
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(
                signal, investment_amount, risk_tolerance, features_df
            )
            
            # Generate risk assessment
            risk_assessment = self._assess_risk(features_df, predictions)
            
            # Create recommendation
            recommendation = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': float(features_df['Close'].iloc[-1]),
                'signal': signal.signal.value,
                'confidence': signal.confidence,
                'price_target': signal.price_target,
                'stop_loss': signal.stop_loss,
                'reasoning': signal.reasoning,
                'risk_level': signal.risk_level,
                'time_horizon': signal.time_horizon,
                'position_sizing': position_sizing,
                'risk_assessment': risk_assessment,
                'technical_analysis': technical_analysis,
                'model_predictions': predictions
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_model_predictions(self, features_df: pd.DataFrame, 
                              models: Dict[str, Any]) -> Dict[str, Any]:
        """Get predictions from all available models"""
        
        predictions = {}
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_next_days'):
                    # For LSTM, get next 5 days prediction
                    pred = model.predict_next_days(features_df, days=5)
                    predictions[model_name] = {
                        'next_day': float(pred[0]),
                        'next_5_days': pred.tolist(),
                        'trend': 'up' if pred[0] > features_df['Close'].iloc[-1] else 'down'
                    }
                else:
                    # For other models, get current prediction
                    pred = model.predict(features_df)
                    if len(pred) > 0:
                        predictions[model_name] = {
                            'prediction': float(pred[-1]),
                            'trend': 'up' if pred[-1] > features_df['Close'].iloc[-1] else 'down'
                        }
            except Exception as e:
                logger.warning(f"Failed to get prediction from {model_name}: {str(e)}")
        
        return predictions
    
    def _analyze_technical_indicators(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators for trading signals"""
        
        latest = features_df.iloc[-1]
        
        analysis = {
            'rsi': {
                'value': latest.get('rsi', 50),
                'signal': 'oversold' if latest.get('rsi', 50) < 30 else 'overbought' if latest.get('rsi', 50) > 70 else 'neutral'
            },
            'macd': {
                'value': latest.get('macd', 0),
                'signal': 'bullish' if latest.get('macd', 0) > 0 else 'bearish'
            },
            'bollinger_bands': {
                'position': latest.get('bb_position', 0.5),
                'signal': 'oversold' if latest.get('bb_position', 0.5) < 0.2 else 'overbought' if latest.get('bb_position', 0.5) > 0.8 else 'neutral'
            },
            'moving_averages': {
                'price_vs_ma20': latest.get('price_vs_ma20', 0),
                'price_vs_ma50': latest.get('price_vs_ma50', 0),
                'signal': 'bullish' if latest.get('price_vs_ma20', 0) > 0 and latest.get('price_vs_ma50', 0) > 0 else 'bearish'
            },
            'volume': {
                'volume_ratio': latest.get('volume_ratio', 1.0),
                'signal': 'high' if latest.get('volume_ratio', 1.0) > 1.5 else 'low' if latest.get('volume_ratio', 1.0) < 0.5 else 'normal'
            }
        }
        
        return analysis
    
    def _generate_trading_signal(self, predictions: Dict[str, Any], 
                                technical_analysis: Dict[str, Any],
                                features_df: pd.DataFrame,
                                investment_amount: float,
                                risk_tolerance: str) -> TradingSignal:
        """Generate trading signal based on predictions and technical analysis"""
        
        current_price = features_df['Close'].iloc[-1]
        reasoning = []
        
        # Count bullish vs bearish signals
        bullish_count = 0
        bearish_count = 0
        
        # Model predictions
        for model_name, pred in predictions.items():
            if 'trend' in pred:
                if pred['trend'] == 'up':
                    bullish_count += 1
                    reasoning.append(f"{model_name.upper()} predicts upward movement")
                else:
                    bearish_count += 1
                    reasoning.append(f"{model_name.upper()} predicts downward movement")
        
        # Technical analysis signals
        tech_signals = []
        
        # RSI
        if technical_analysis['rsi']['signal'] == 'oversold':
            bullish_count += 1
            tech_signals.append("RSI indicates oversold conditions")
        elif technical_analysis['rsi']['signal'] == 'overbought':
            bearish_count += 1
            tech_signals.append("RSI indicates overbought conditions")
        
        # MACD
        if technical_analysis['macd']['signal'] == 'bullish':
            bullish_count += 1
            tech_signals.append("MACD shows bullish momentum")
        else:
            bearish_count += 1
            tech_signals.append("MACD shows bearish momentum")
        
        # Bollinger Bands
        if technical_analysis['bollinger_bands']['signal'] == 'oversold':
            bullish_count += 1
            tech_signals.append("Price near lower Bollinger Band")
        elif technical_analysis['bollinger_bands']['signal'] == 'overbought':
            bearish_count += 1
            tech_signals.append("Price near upper Bollinger Band")
        
        # Moving Averages
        if technical_analysis['moving_averages']['signal'] == 'bullish':
            bullish_count += 1
            tech_signals.append("Price above moving averages")
        else:
            bearish_count += 1
            tech_signals.append("Price below moving averages")
        
        reasoning.extend(tech_signals)
        
        # Determine signal
        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            signal_type = SignalType.HOLD
            confidence = 0.5
        else:
            bullish_ratio = bullish_count / total_signals
            
            if bullish_ratio > 0.6:
                signal_type = SignalType.BUY
                confidence = min(0.9, 0.5 + bullish_ratio * 0.4)
            elif bullish_ratio < 0.4:
                signal_type = SignalType.SELL
                confidence = min(0.9, 0.5 + (1 - bullish_ratio) * 0.4)
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
        
        # Calculate price targets
        if signal_type == SignalType.BUY:
            # Conservative 5% upside target
            price_target = current_price * 1.05
            stop_loss = current_price * 0.95
            risk_level = "moderate" if confidence < 0.7 else "low"
            time_horizon = "1-3 months"
        elif signal_type == SignalType.SELL:
            # Conservative 5% downside target
            price_target = current_price * 0.95
            stop_loss = current_price * 1.05
            risk_level = "high" if confidence < 0.7 else "moderate"
            time_horizon = "1-3 months"
        else:
            price_target = current_price
            stop_loss = current_price * 0.98
            risk_level = "low"
            time_horizon = "wait and watch"
        
        return TradingSignal(
            signal=signal_type,
            confidence=confidence,
            price_target=price_target,
            stop_loss=stop_loss,
            reasoning=reasoning,
            risk_level=risk_level,
            time_horizon=time_horizon
        )
    
    def _calculate_position_sizing(self, signal: TradingSignal, 
                                  investment_amount: float,
                                  risk_tolerance: str,
                                  features_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate optimal position size based on risk management"""
        
        current_price = features_df['Close'].iloc[-1]
        
        # Risk per trade (1-2% of portfolio)
        risk_percentage = 0.02 if risk_tolerance == "aggressive" else 0.01
        
        # Calculate position size based on stop loss
        risk_per_share = abs(current_price - signal.stop_loss)
        max_risk_amount = investment_amount * risk_percentage
        max_shares = max_risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # Adjust for confidence
        confidence_multiplier = signal.confidence
        recommended_shares = int(max_shares * confidence_multiplier)
        
        # Calculate position value
        position_value = recommended_shares * current_price
        position_percentage = (position_value / investment_amount) * 100
        
        return {
            'recommended_shares': recommended_shares,
            'position_value': position_value,
            'position_percentage': position_percentage,
            'risk_amount': max_risk_amount,
            'risk_per_share': risk_per_share
        }
    
    def _assess_risk(self, features_df: pd.DataFrame, 
                    predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall investment risk"""
        
        latest = features_df.iloc[-1]
        
        # Volatility risk
        volatility_20 = latest.get('volatility_20', 0)
        volatility_risk = "high" if volatility_20 > 0.3 else "moderate" if volatility_20 > 0.2 else "low"
        
        # Market risk (using beta if available)
        beta = latest.get('beta', 1.0)
        market_risk = "high" if beta > 1.5 else "moderate" if beta > 0.8 else "low"
        
        # Prediction consistency
        if predictions:
            trends = [pred.get('trend', 'neutral') for pred in predictions.values()]
            consistency = len(set(trends)) / len(trends) if trends else 1.0
            prediction_risk = "high" if consistency < 0.5 else "moderate" if consistency < 0.8 else "low"
        else:
            prediction_risk = "high"
        
        # Overall risk score
        risk_scores = {
            'low': 1,
            'moderate': 2,
            'high': 3
        }
        
        total_risk_score = (
            risk_scores.get(volatility_risk, 2) +
            risk_scores.get(market_risk, 2) +
            risk_scores.get(prediction_risk, 2)
        ) / 3
        
        overall_risk = "high" if total_risk_score > 2.5 else "moderate" if total_risk_score > 1.5 else "low"
        
        return {
            'volatility_risk': volatility_risk,
            'market_risk': market_risk,
            'prediction_risk': prediction_risk,
            'overall_risk': overall_risk,
            'risk_score': total_risk_score
        }
    
    def get_portfolio_recommendations(self, symbols: List[str], 
                                    investment_amount: float = 100000) -> Dict[str, Any]:
        """Get recommendations for multiple stocks (portfolio)"""
        
        recommendations = {}
        total_allocated = 0
        
        for symbol in symbols:
            try:
                # Allocate equal amount to each stock initially
                allocation = investment_amount / len(symbols)
                rec = self.get_recommendation(symbol, allocation)
                recommendations[symbol] = rec
                
                if 'position_sizing' in rec:
                    total_allocated += rec['position_sizing']['position_value']
                    
            except Exception as e:
                logger.error(f"Failed to get recommendation for {symbol}: {str(e)}")
                recommendations[symbol] = {'error': str(e)}
        
        # Portfolio summary
        portfolio_summary = {
            'total_investment': investment_amount,
            'total_allocated': total_allocated,
            'allocation_percentage': (total_allocated / investment_amount) * 100,
            'recommendations': recommendations
        }
        
        return portfolio_summary

def main():
    """Test the investment advisor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Get investment recommendations')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--amount', type=float, default=10000, help='Investment amount')
    parser.add_argument('--risk', type=str, default='moderate', choices=['low', 'moderate', 'aggressive'])
    
    args = parser.parse_args()
    
    advisor = InvestmentAdvisor()
    recommendation = advisor.get_recommendation(args.symbol, args.amount, args.risk)
    
    if 'error' in recommendation:
        print(f"Error: {recommendation['error']}")
        return 1
    
    # Print recommendation
    print(f"\n{'='*60}")
    print(f"INVESTMENT RECOMMENDATION FOR {args.symbol}")
    print(f"{'='*60}")
    print(f"Current Price: ${recommendation['current_price']:.2f}")
    print(f"Signal: {recommendation['signal']}")
    print(f"Confidence: {recommendation['confidence']:.1%}")
    print(f"Price Target: ${recommendation['price_target']:.2f}")
    print(f"Stop Loss: ${recommendation['stop_loss']:.2f}")
    print(f"Risk Level: {recommendation['risk_level']}")
    print(f"Time Horizon: {recommendation['time_horizon']}")
    
    print(f"\nPosition Sizing:")
    pos = recommendation['position_sizing']
    print(f"  Recommended Shares: {pos['recommended_shares']}")
    print(f"  Position Value: ${pos['position_value']:.2f}")
    print(f"  Portfolio Allocation: {pos['position_percentage']:.1f}%")
    
    print(f"\nRisk Assessment:")
    risk = recommendation['risk_assessment']
    print(f"  Volatility Risk: {risk['volatility_risk']}")
    print(f"  Market Risk: {risk['market_risk']}")
    print(f"  Prediction Risk: {risk['prediction_risk']}")
    print(f"  Overall Risk: {risk['overall_risk']}")
    
    print(f"\nReasoning:")
    for i, reason in enumerate(recommendation['reasoning'], 1):
        print(f"  {i}. {reason}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 