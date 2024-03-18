import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time

# Global variable for caching data
data_cache = {}

def get_live_stock_data(symbol, interval='1m', period='1d'):
    try:
        # Check if data is already in cache
        if symbol in data_cache and interval in data_cache[symbol]:
            return data_cache[symbol][interval]

        # Fetch real-time data
        stock_data = yf.download(symbol, interval=interval, period=period)

        # Cache the data
        if symbol not in data_cache:
            data_cache[symbol] = {}
        data_cache[symbol][interval] = stock_data

        return stock_data
    except Exception as e:
        st.error(f"Error retrieving data for {symbol}: {e}")
        return None

def save_to_csv(data, filename='stock_data.csv'):
    data.to_csv(filename, index=True)

def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    return fig

def main():
    st.title("Real-Time Stock Data Visualization")

    symbol = st.text_input("Enter Stock Ticker Symbol:", 'AAPL')
    interval = st.selectbox("Select Data Interval:", ['1m', '5m', '15m', '1h'])
    period = st.selectbox("Select Data Period:", ['1d', '5d', '1mo', '3mo'])

    # Throttle requests to avoid violating usage policies
    time.sleep(2)

    intraday_data = get_live_stock_data(symbol, interval=interval, period=period)

    if intraday_data is not None:
        st.subheader(f"Real-Time Stock Data for {symbol}")
        st.dataframe(intraday_data)

        # Save data to CSV
        if st.button("Save Data to CSV"):
            save_to_csv(intraday_data)
            st.success(f"Data saved to stock_data.csv")

        # Plot candlestick chart
        st.subheader("Candlestick Chart")
        fig = plot_candlestick(intraday_data)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
