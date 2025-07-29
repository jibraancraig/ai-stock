# Stock Market AI - Installation Guide

## Quick Start

1. **Install Dependencies:**

   ```bash
   pip3 install -r requirements.txt
   ```

2. **Run Setup Script:**

   ```bash
   python3 setup.py
   ```

3. **Test the System:**
   ```bash
   python3 quick_start.py
   ```

## Manual Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Install Dependencies

```bash
# Upgrade pip
pip3 install --upgrade pip

# Install required packages
pip3 install numpy pandas scikit-learn tensorflow keras yfinance matplotlib seaborn plotly dash dash-bootstrap-components ta scipy statsmodels xgboost lightgbm flask requests python-dotenv joblib
```

### Step 2: Test Installation

```bash
# Test data collection
python3 data_collector.py --symbol AAPL --period 1mo --save

# Test feature engineering
python3 feature_engineering.py
```

## Usage Examples

### 1. Data Collection

```bash
# Collect stock data
python3 data_collector.py --symbol AAPL --period 2y --save

# Collect data for multiple stocks
python3 data_collector.py --symbol TSLA --period 1y --save
```

### 2. Model Training

```bash
# Train all models for a stock
python3 train_models.py --symbol AAPL --period 2y

# View training results
cat training_summary.json
```

### 3. Get Predictions

```bash
# Get quick predictions
python3 predict.py --symbol AAPL --days 5

# Get investment recommendations
python3 investment_advisor.py --symbol AAPL --amount 10000 --risk moderate
```

### 4. Web Dashboard

```bash
# Launch interactive dashboard
python3 app.py

# Open browser to: http://127.0.0.1:8050
```

## Project Structure

```
stock_ai_project/
├── data_collector.py      # Stock data fetching
├── feature_engineering.py # Technical indicators & features
├── models/               # ML model implementations
│   ├── lstm_model.py
│   ├── random_forest_model.py
│   ├── xgboost_model.py
│   ├── svm_model.py
│   └── ensemble_model.py
├── train_models.py       # Model training pipeline
├── predict.py           # Quick predictions
├── investment_advisor.py # Buy/sell recommendations
├── app.py               # Web dashboard
├── quick_start.py       # Demo script
├── setup.py             # Installation script
├── requirements.txt     # Dependencies
└── data/                # Stored data and models
```

## Features

### Data Collection

- Real-time and historical stock data
- Multiple time periods and intervals
- Automatic data validation and cleaning

### Feature Engineering

- 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Mathematical features (momentum, volatility, etc.)
- Calculus-based features (derivatives, integrals)
- Statistical features (z-scores, percentiles, etc.)
- Pattern recognition features

### Machine Learning Models

- **LSTM Neural Networks**: Time series prediction
- **Random Forest**: Ensemble learning with feature importance
- **XGBoost**: Gradient boosting with hyperparameter tuning
- **SVM**: Support Vector Machines for regression
- **Ensemble Model**: Combines all models for better accuracy

### Investment Advisor

- Buy/Sell/Hold recommendations
- Confidence scores and reasoning
- Position sizing and risk management
- Technical analysis integration
- Portfolio recommendations

### Web Dashboard

- Interactive charts and visualizations
- Real-time analysis
- Technical indicators display
- Risk assessment visualization

## Mathematical Foundation

The system uses advanced mathematical concepts:

1. **Technical Analysis**: Price patterns, momentum, volatility indicators
2. **Statistical Modeling**: Time series analysis, regression, correlation
3. **Calculus**: Rate of change, derivatives for trend analysis
4. **Probability Theory**: Risk assessment and confidence intervals
5. **Machine Learning**: Neural networks, ensemble methods, feature selection

## Risk Disclaimer

⚠️ **IMPORTANT**: This is for educational purposes only.

- Always do your own research before making investment decisions
- Past performance does not guarantee future results
- Consider consulting with financial advisors
- Never invest more than you can afford to lose
- The system provides recommendations, not financial advice

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed

   ```bash
   pip3 install -r requirements.txt
   ```

2. **Data Collection Errors**: Check internet connection and stock symbol validity

3. **Model Training Errors**: Ensure sufficient data (at least 6 months recommended)

4. **Memory Issues**: Reduce batch sizes or use smaller datasets

### Getting Help

1. Check the logs for detailed error messages
2. Verify Python version (3.8+ required)
3. Ensure all dependencies are properly installed
4. Test with a simple stock symbol first (e.g., AAPL)

## Performance Tips

1. **For Training**: Use GPU if available for faster LSTM training
2. **For Prediction**: Models are cached after training for faster inference
3. **For Data**: Use longer periods (1-2 years) for better model accuracy
4. **For Features**: The system automatically selects the most relevant features

## Advanced Usage

### Custom Models

You can extend the system by adding new models in the `models/` directory.

### Custom Features

Add new technical indicators in `feature_engineering.py`.

### API Integration

The system can be integrated with trading APIs for automated trading (advanced users only).

### Backtesting

Use historical data to test strategy performance before live trading.
