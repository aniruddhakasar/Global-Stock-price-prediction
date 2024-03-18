import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime
import time

# Global variable for caching data
data_cache = {}

def get_live_stock_data(symbol, interval='5m', period='1d'):
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

def plot_candlestick(data, symbol):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         increasing_line_color='green',
                                         decreasing_line_color='red')])

    fig.update_layout(
        title=f'{symbol} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
    )

    # Determine the y-axis range based on the stock price
    stock_price = data['Close'].max()
    if stock_price <100:
        y_axis_range = 5
    elif 200 <= stock_price < 300:
        y_axis_range = 20
    else:
        y_axis_range = 50

    # Set the y-axis range dynamically
    fig.update_yaxes(range=[data['Low'].min() - y_axis_range, data['High'].max() + y_axis_range])

    return fig

def main():
    st.title("Stock Data Visualization")

    symbol_options = {
        "TATAMOTORS.NS": "1995-12-25",
        "TATAPOWER.NS": "1996-01-01",
        "TATASTEEL.NS": "1996-01-01",
        "TCS.NS": "F2002-08-12",
        "TATACHEM.NS": "1996-01-01",
        "TATACONSUM.NS":"1996-01-01",
        "TITAN.NS":"1996-01-01",
        "TATAELXSI.NS":"2002-07-01",
        "RALLIS.NS":"2002-07-01",
        "TATACOFFEE.NS":"2002-07-01",
    }

    # User input for stock symbol, number of days, and existing CSV file
    ticker_key = st.session_state.ticker_key if 'ticker_key' in st.session_state else 0
    ticker_key += 1
    st.session_state.ticker_key = ticker_key

    ticker = st.sidebar.selectbox(f'Ticker {ticker_key} :', list(symbol_options.keys()))
    selected_time_frame = st.sidebar.radio('Select a time frame:', ["1 Day", "1 Week", "1 Month", "1 Year", "5 Years", "10 Years"])

    # Map the selected time frame to its duration in days
    duration_mapping = {
        "1 Day": 1,
        "1 Week": 7,
        "1 Month": 30,
        "1 Year": 365,
        "5 Years": 5 * 365,
        "10 Years": 10 * 365
    }

    # Calculate start and end dates based on the selected time frame
    end_date = datetime.date.today()  # End date is today
    duration = duration_mapping.get(selected_time_frame, 1)  # Default to 1 day if not found
    start_date = end_date - datetime.timedelta(days=duration)

    # Check if the selected time frame is "1 Day" to use live data
    if selected_time_frame == "1 Day":
        interval = '1m'
        period = '1d'
        data = get_live_stock_data(ticker, interval=interval, period=period)
    else:
        data = yf.download(ticker, start=start_date, end=end_date)

    if data is not None:
        st.subheader(f"Stock Data for {ticker}")
        st.dataframe(data)

        # Save data to CSV
        if st.button("Save Data to CSV"):
            save_to_csv(data)
            st.success(f"Data saved to stock_data.csv")

        # Plot candlestick chart
        st.subheader("Candlestick Chart")
        fig = plot_candlestick(data, ticker)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
