import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Function to get intraday data
def get_intraday_data(ticker, interval='1m'):
    return yf.download(tickers=ticker, interval=interval)

# Function to generate candlestick plot
def plot_candlestick(data, interval):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Candlestick'), row=1, col=1)

    fig.update_layout(title=f'{interval.capitalize()} Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Price')

    fig.show()

# Main function
def main():
    ticker = input("Enter stock symbol: ")
    interval = input("Enter interval (1m, 15m, 1h): ")

    # Get intraday data
    data = get_intraday_data(ticker, interval)

    if not data.empty:
        print("Intraday data collected successfully.")

        # Plot candlestick chart
        plot_candlestick(data, interval)

    else:
        print("Failed to collect intraday data.")

if __name__ == "__main__":
    main()
