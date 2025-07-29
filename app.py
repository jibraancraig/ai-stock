#!/usr/bin/env python3
"""
Production-ready Stock Market AI Dashboard
Deployable to Heroku, Railway, or other cloud platforms
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import os

# Custom CSS for exact replication of the design
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Montserrat', sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a0f0f 25%, #2d1b1b 50%, #3d2b2b 75%, #4d3b3b 100%);
    color: #ffffff;
    min-height: 100vh;
    overflow-x: hidden;
}

.dash-container {
    background: transparent;
    padding: 0;
}

/* Header Styling */
.header-container {
    text-align: center;
    padding: 2rem 0 1rem 0;
    position: relative;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #8b0000;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.creator-signature {
    position: absolute;
    top: 1rem;
    right: 2rem;
    font-size: 0.875rem;
    color: rgba(255, 255, 255, 0.7);
    font-weight: 400;
}

/* Search Section */
.search-section {
    background: rgba(45, 27, 27, 0.8);
    border: 1px solid rgba(139, 0, 0, 0.4);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 0 2rem 2rem 2rem;
    box-shadow: 0 8px 32px rgba(139, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

.search-title {
    color: #8b0000;
    font-weight: 600;
    font-size: 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.search-input-group {
    display: flex;
    gap: 1rem;
    align-items: center;
}

.search-input {
    flex: 1;
    background: rgba(45, 27, 27, 0.9);
    border: 1px solid rgba(139, 0, 0, 0.5);
    border-radius: 10px;
    color: #ffffff;
    font-weight: 400;
    padding: 0.75rem 1rem;
    font-size: 1rem;
}

.search-input:focus {
    outline: none;
    border-color: #8b0000;
    box-shadow: 0 0 0 2px rgba(139, 0, 0, 0.25);
    color: #ffffff;
}

.search-input::placeholder {
    color: rgba(255, 255, 255, 0.6);
}

.analyze-btn {
    background: #8b0000;
    border: none;
    border-radius: 10px;
    color: #ffffff;
    font-weight: 500;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    white-space: nowrap;
}

.analyze-btn:hover {
    background: #a00000;
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(139, 0, 0, 0.4);
}

/* Cards Grid */
.cards-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    padding: 0 2rem 2rem 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(139, 0, 0, 0.3);
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(139, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    min-height: 300px;
    display: flex;
    flex-direction: column;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(139, 0, 0, 0.3);
}

.card-header {
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(139, 0, 0, 0.2);
}

.card-title {
    color: #8b0000;
    font-weight: 600;
    font-size: 1.25rem;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-body {
    padding: 1.5rem;
    flex: 1;
    display: flex;
    flex-direction: column;
}

.card-content {
    flex: 1;
}

/* Typography */
.bold-number {
    font-weight: 700;
    color: #8b0000;
}

.bold-text {
    font-weight: 600;
    color: #000000;
}

.regular-text {
    color: #000000;
    font-weight: 400;
    margin-bottom: 0.75rem;
}

/* Alert Styling */
.alert {
    border-radius: 10px;
    border: none;
    font-weight: 500;
    padding: 1rem;
    margin-bottom: 1rem;
}

.alert-success {
    background: rgba(40, 167, 69, 0.2);
    color: #4ade80;
    border-left: 4px solid #4ade80;
}

.alert-danger {
    background: rgba(220, 53, 69, 0.2);
    color: #f87171;
    border-left: 4px solid #f87171;
}

.alert-warning {
    background: rgba(255, 193, 7, 0.2);
    color: #fbbf24;
    border-left: 4px solid #fbbf24;
}

/* List Styling */
ul {
    list-style: none;
    padding-left: 0;
}

li {
    color: #000000;
    margin-bottom: 0.5rem;
    padding-left: 1rem;
    position: relative;
}

li:before {
    content: "•";
    color: #8b0000;
    font-weight: bold;
    position: absolute;
    left: 0;
}

/* Chart Styling */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

.js-plotly-plot .plotly .bglayer rect {
    fill: transparent !important;
}

/* Loading Animation */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(139, 0, 0, 0.3);
    border-radius: 50%;
    border-top-color: #8b0000;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 768px) {
    .cards-grid {
        grid-template-columns: 1fr;
        gap: 1.5rem;
        padding: 0 1rem 1rem 1rem;
    }
    
    .search-section {
        margin: 0 1rem 1.5rem 1rem;
    }
    
    .search-input-group {
        flex-direction: column;
        gap: 0.75rem;
    }
    
    .main-title {
        font-size: 2rem;
    }
    
    .creator-signature {
        position: static;
        text-align: center;
        margin-top: 0.5rem;
    }
}
"""

# Initialize the Dash app with custom CSS
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Machine Learning AI Investor"

# Add custom CSS
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{app.title}</title>
        <style>{CUSTOM_CSS}</style>
        {{%metas%}}
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📈 Machine Learning AI Investor", className="main-title"),
        html.Div("Made by Jibraan Craig", className="creator-signature")
    ], className="header-container"),
    
    # Search Section
    html.Div([
        html.H5("🔍 Stock Analysis", className="search-title"),
        html.Div([
            dcc.Input(
                id="symbol-input",
                placeholder="Please search any stock symbol (e.g AAPL)",
                value="AAPL",
                type="text",
                className="search-input"
            ),
            html.Button("Analyse", id="analyze-btn", className="analyze-btn")
        ], className="search-input-group")
    ], className="search-section"),
    
    # Loading Output
    html.Div(id="loading-output"),
    
    # Cards Grid
    html.Div([
        # Top Left Card - Stock Info
        html.Div([
            html.Div([
                html.H5(id="stock-title", children="📊 AAPL - Apple Inc.", className="card-title")
            ], className="card-header"),
            html.Div(id="stock-info-card", className="card-body")
        ], className="card"),
        
        # Top Right Card - Trading Recommendation
        html.Div([
            html.Div([
                html.H5("🎯 Trading Recommendation", className="card-title")
            ], className="card-header"),
            html.Div(id="recommendation-card", className="card-body")
        ], className="card"),
        
        # Bottom Left Card - Technical Analysis
        html.Div([
            html.Div([
                html.H5("📈 Technical Analysis", className="card-title")
            ], className="card-header"),
            html.Div(id="technical-content", className="card-body")
        ], className="card"),
        
        # Bottom Right Card - Risk Assessment
        html.Div([
            html.Div([
                html.H5("⚠️ Risk Assessment", className="card-title")
            ], className="card-header"),
            html.Div(id="risk-content", className="card-body")
        ], className="card")
    ], className="cards-grid"),
    
    # Hidden chart div (will be shown when needed)
    html.Div(id="price-chart", style={'display': 'none'})
], style={'min-height': '100vh'})

def fetch_stock_data(symbol, period="6mo"):
    """Fetch stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        return None

def add_technical_indicators(df):
    """Add basic technical indicators"""
    try:
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        
        # Moving Averages
        df['MA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['MA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        
        return df
    except:
        return df

def generate_recommendation(df):
    """Generate simple trading recommendation"""
    if len(df) < 20:
        return "Insufficient data for analysis"
    
    latest = df.iloc[-1]
    prev_20 = df.iloc[-20]
    
    # Price trend
    price_change = ((latest['Close'] - prev_20['Close']) / prev_20['Close']) * 100
    
    # RSI analysis
    rsi = latest.get('RSI', 50)
    
    # MACD analysis
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    
    # Volume analysis
    avg_volume = df['Volume'].mean()
    current_volume = latest['Volume']
    volume_ratio = current_volume / avg_volume
    
    # Generate recommendation
    signals = []
    confidence = 0
    
    if price_change > 5:
        signals.append("Strong uptrend (+{:.1f}%)".format(price_change))
        confidence += 25
    elif price_change > 0:
        signals.append("Slight uptrend (+{:.1f}%)".format(price_change))
        confidence += 15
    elif price_change < -5:
        signals.append("Strong downtrend ({:.1f}%)".format(price_change))
        confidence -= 25
    else:
        signals.append("Sideways movement ({:.1f}%)".format(price_change))
        confidence += 5
    
    if rsi < 30:
        signals.append("Oversold (RSI: {:.1f})".format(rsi))
        confidence += 20
    elif rsi > 70:
        signals.append("Overbought (RSI: {:.1f})".format(rsi))
        confidence -= 20
    
    if macd > macd_signal:
        signals.append("MACD bullish")
        confidence += 15
    else:
        signals.append("MACD bearish")
        confidence -= 15
    
    if volume_ratio > 1.5:
        signals.append("High volume ({:.1f}x average)".format(volume_ratio))
        confidence += 10
    
    # Determine recommendation
    if confidence > 30:
        recommendation = "BUY"
        color = "success"
    elif confidence < -30:
        recommendation = "SELL"
        color = "danger"
    else:
        recommendation = "HOLD"
        color = "warning"
    
    return {
        'recommendation': recommendation,
        'confidence': abs(confidence),
        'signals': signals,
        'color': color,
        'price_change': price_change,
        'rsi': rsi,
        'volume_ratio': volume_ratio
    }

@app.callback(
    [Output("loading-output", "children"),
     Output("stock-info-card", "children"),
     Output("recommendation-card", "children"),
     Output("technical-content", "children"),
     Output("risk-content", "children")],
    [Input("analyze-btn", "n_clicks")],
    [State("symbol-input", "value")]
)
def update_analysis(n_clicks, symbol):
    if n_clicks == 0:
        return "", "", "", "", ""
    
    if not symbol:
        return html.Div([
            html.Div(className="loading-spinner"),
            " Please enter a stock symbol"
        ], style={'textAlign': 'center', 'color': '#ff6b6b'}), "", "", "", ""
    
    try:
        # Fetch data
        df = fetch_stock_data(symbol.upper())
        if df is None or len(df) == 0:
            return html.Div([
                html.Div(className="loading-spinner"),
                f" Error fetching data for {symbol}"
            ], style={'textAlign': 'center', 'color': '#f87171'}), "", "", "", ""
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Get stock info
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        # Stock info card content
        stock_info = html.Div([
            html.P([
                "💰 Current Price: ",
                html.Span(f"${df['Close'].iloc[-1]:.2f}", className="bold-number")
            ], className="regular-text"),
            html.P([
                "📈 Market Cap: ",
                html.Span(f"${info.get('marketCap', 0):,.0f}", className="bold-number")
            ], className="regular-text"),
            html.P([
                "🏢 Sector: ",
                html.Span(info.get('sector', 'N/A'), className="bold-text")
            ], className="regular-text"),
            html.P([
                "📅 Data Points: ",
                html.Span(f"{len(df)} days", className="bold-number")
            ], className="regular-text"),
            html.P([
                "📊 Volume: ",
                html.Span(f"{df['Volume'].iloc[-1]:,.0f}", className="bold-number")
            ], className="regular-text")
        ], className="card-content")
        
        # Generate recommendation
        rec = generate_recommendation(df)
        
        recommendation_card = html.Div([
            html.Div(
                f"{rec['recommendation']} - Confidence: {rec['confidence']:.0f}%",
                className=f"alert alert-{rec['color']}"
            ),
            html.H6("Key Signals:", className="bold-text", style={'marginTop': '1rem', 'marginBottom': '0.5rem'}),
            html.Ul([html.Li(signal) for signal in rec['signals']])
        ], className="card-content")
        
        # Technical analysis
        technical_content = html.Div([
            html.P([
                "RSI: ",
                html.Span(f"{rec['rsi']:.1f}", className="bold-number")
            ], className="regular-text"),
            html.P([
                "Price Change (20d): ",
                html.Span(f"{rec['price_change']:+.1f}%", className="bold-number")
            ], className="regular-text"),
            html.P([
                "Volume Ratio: ",
                html.Span(f"{rec['volume_ratio']:.1f}x average", className="bold-number")
            ], className="regular-text"),
            html.P([
                "MACD: ",
                html.Span(f"{df.iloc[-1].get('MACD', 0):.3f}", className="bold-number")
            ], className="regular-text"),
            html.P([
                "Data Quality: ",
                html.Span(f"Good ({len(df)} days)", className="bold-text")
            ], className="regular-text")
        ], className="card-content")
        
        # Risk assessment
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
        if volatility > 30:
            risk_level = "High"
            risk_color = "danger"
        elif volatility > 20:
            risk_level = "Moderate"
            risk_color = "warning"
        else:
            risk_level = "Low"
            risk_color = "success"
        
        risk_content = html.Div([
            html.Div(f"Risk Level: {risk_level}", className=f"alert alert-{risk_color}"),
            html.P([
                "Volatility: ",
                html.Span(f"{volatility:.1f}%", className="bold-number")
            ], className="regular-text"),
            html.P([
                "Max Drawdown: ",
                html.Span(f"{((df['Close'] / df['Close'].expanding().max()) - 1).min() * 100:.1f}%", className="bold-number")
            ], className="regular-text"),
            html.P([
                "Sharpe Ratio: ",
                html.Span(f"{df['Close'].pct_change().mean() / df['Close'].pct_change().std() * np.sqrt(252):.2f}", className="bold-number")
            ], className="regular-text"),
            html.P([
                "Beta: ",
                html.Span("N/A", className="bold-text")
            ], className="regular-text")
        ], className="card-content")
        
        return "", stock_info, recommendation_card, technical_content, risk_content
        
    except Exception as e:
        return html.Div([
            html.Div(className="loading-spinner"),
            f" Error: {str(e)}"
        ], style={'textAlign': 'center', 'color': '#f87171'}), "", "", "", ""

# Callback to update stock title
@app.callback(
    Output("stock-title", "children"),
    [Input("analyze-btn", "n_clicks")],
    [State("symbol-input", "value")]
)
def update_stock_title(n_clicks, symbol):
    if n_clicks and symbol:
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            company_name = info.get('longName', symbol.upper())
            return f"📊 {symbol.upper()} - {company_name}"
        except:
            return f"📊 {symbol.upper()}"
    return "📊 AAPL - Apple Inc."

# Production server configuration
if __name__ == '__main__':
    # Get port from environment variable (for Heroku) or use default
    port = int(os.environ.get('PORT', 8080))
    
    print("🚀 Starting Stock Market AI Dashboard...")
    print(f"📊 Open your browser and go to: http://127.0.0.1:{port}")
    print("🔍 Enter a stock symbol (e.g., AAPL, MSFT, GOOGL) and click Analyze")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Use production server for deployment
    app.run(debug=False, host='0.0.0.0', port=port) 