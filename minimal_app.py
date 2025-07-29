#!/usr/bin/env python3
"""
Minimal Stock Market AI Web Dashboard
Simplified version that should work reliably
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
import ta

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Stock Market AI"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📈 Stock Market AI", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔍 Stock Analysis", className="card-title"),
                    dbc.InputGroup([
                        dbc.Input(id="symbol-input", placeholder="Enter stock symbol (e.g., AAPL)", 
                                 value="AAPL", type="text"),
                        dbc.Button("Analyze", id="analyze-btn", color="primary", n_clicks=0)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="results")
        ], width=12)
    ])
], fluid=True)

@app.callback(
    Output("results", "children"),
    [Input("analyze-btn", "n_clicks")],
    [State("symbol-input", "value")]
)
def update_analysis(n_clicks, symbol):
    if n_clicks == 0 or not symbol:
        return ""
    
    try:
        # Fetch data
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(period="6mo")
        data.reset_index(inplace=True)
        
        if len(data) == 0:
            return dbc.Alert(f"Error: No data found for {symbol}", color="danger")
        
        # Add basic indicators
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['MA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        
        latest = data.iloc[-1]
        prev_20 = data.iloc[-20] if len(data) >= 20 else data.iloc[0]
        
        # Calculate metrics
        price_change = ((latest['Close'] - prev_20['Close']) / prev_20['Close']) * 100
        rsi = latest.get('RSI', 50)
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        
        # Generate recommendation
        if price_change > 5 and rsi < 70:
            recommendation = "BUY"
            color = "success"
        elif price_change < -5 and rsi > 30:
            recommendation = "SELL"
            color = "danger"
        else:
            recommendation = "HOLD"
            color = "warning"
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        if 'MA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['Date'], y=data['MA_20'],
                mode='lines', name='MA 20',
                line=dict(color='orange')
            ))
        
        fig.update_layout(
            title=f'{symbol.upper()} Stock Price',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400
        )
        
        return [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"📊 {symbol.upper()} Analysis", className="card-title"),
                            html.P(f"💰 Current Price: ${latest['Close']:.2f}"),
                            html.P(f"📈 20-Day Change: {price_change:+.2f}%"),
                            html.P(f"📊 RSI: {rsi:.1f}"),
                            html.P(f"⚠️ Volatility: {volatility:.1f}%")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("🎯 Recommendation", className="card-title"),
                            dbc.Alert(
                                f"{recommendation}",
                                color=color,
                                className="mb-3"
                            ),
                            html.P(f"Trend: {'Bullish' if price_change > 0 else 'Bearish'}"),
                            html.P(f"Risk: {'High' if volatility > 30 else 'Moderate' if volatility > 20 else 'Low'}")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig)
                ], width=12)
            ])
        ]
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger")

if __name__ == '__main__':
    print("🚀 Starting Stock Market AI Dashboard...")
    print("📊 Open your browser and go to: http://127.0.0.1:8080")
    print("🔍 Enter a stock symbol and click Analyze")
    print("⏹️  Press Ctrl+C to stop the server")
    app.run(debug=False, host='127.0.0.1', port=3000) 