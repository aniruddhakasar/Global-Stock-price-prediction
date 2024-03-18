# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Stock Analysis Dashboard"),

    # Input for stock symbol
    dcc.Input(id='stock-input', type='text', value='AAPL', placeholder='Enter stock symbol'),

    # Dropdown for selecting time period
    dcc.Dropdown(
        id='time-period',
        options=[
            {'label': '1y', 'value': '1y'},
            {'label': '2y', 'value': '2y'},
            {'label': '5y', 'value': '5y'}
        ],
        value='1y',
        style={'width': '50%'}
    ),

    # Candlestick chart for stock prices
    dcc.Graph(id='candlestick-chart'),

    # Line chart for moving average
    dcc.Graph(id='moving-average-chart')
])

# Callback to update charts based on user input
@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('moving-average-chart', 'figure')],
    [Input('stock-input', 'value'),
     Input('time-period', 'value')]
)
def update_charts(stock_symbol, time_period):
    # Fetch stock data using yfinance
    stock_data = yf.download(stock_symbol, period=time_period)

    # Candlestick chart
    candlestick_chart = px.candlestick(stock_data, x=stock_data.index,
                                        open='Open', high='High',
                                        low='Low', close='Close',
                                        title=f'{stock_symbol} Stock Prices')

    # Calculate moving average
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()

    # Line chart for moving average
    moving_average_chart = px.line(stock_data, x=stock_data.index,
                                   y=['Close', 'MA50'],
                                   title=f'{stock_symbol} Stock Prices with 50-day Moving Average')

    return candlestick_chart, moving_average_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
