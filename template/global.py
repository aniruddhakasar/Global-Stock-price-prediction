import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import yfinance as yf
import plotly.express as px
from alpha_vantage.timeseries import TimeSeries
import requests
import plotly.graph_objs as go
import datetime
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Title
st.title("Global Stock Price Prediction")



# Webscrapping through yohoo finance library
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
ticker = st.sidebar.selectbox('ticker:', list(symbol_options.keys()))
start_date = st.sidebar.date_input('Start date')
end_date = st.sidebar.date_input('End date')
data = yf.download(ticker, start=start_date, end=end_date)




# List of time frame options
time_frames = [ "1 month", "1 year", "5 years", "10 years"]

# Create buttons for selecting time frame
selected_time_frame = st.sidebar.radio('Select a time frame:', time_frames)

# Map the selected time frame to its duration in days
duration_mapping = {
    "1 month": 30,
    "1 year": 365,
    "5 years": 5 * 365,
    "10 years": 10 * 365
}

# Calculate start and end dates based on the selected time frame
end_date = datetime.date.today()  # End date is today
duration = duration_mapping.get(selected_time_frame, 1)  # Default to 1 day if not found
start_date = end_date - datetime.timedelta(days=duration)

# Retrieve historical stock data using Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                increasing_line_color='green',
                decreasing_line_color='red')])

fig.update_layout(
    title=f'{ticker} Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price',
)

# Determine the y-axis range based on the stock price
stock_price = data['Close'].max()
if stock_price < 200:
    y_axis_range = 5
elif 200 <= stock_price < 500:
    y_axis_range = 20
else:
    y_axis_range = 50

# Set the y-axis range dynamically
fig.update_yaxes(range=[data['Low'].min() - y_axis_range, data['High'].max() + y_axis_range])

st.plotly_chart(fig)







# Alpha Vantage API Key (Replace with your own API key)
api_key = 'KIEE5XOBEMJXB5LG'

# Initialize the Alpha Vantage TimeSeries API
ts = TimeSeries(key=api_key, output_format='pandas')

# List of stock symbols (Nifty 50 and Bank Nifty)
symbols = ['^NSEI', '^NSEBANK']

st.title('Live Candlestick Chart - Nifty 50 and Bank Nifty')

# User selects a stock symbol
selected_symbol = st.selectbox('Select a stock symbol:', symbols)

if st.button('Show Live Candlestick Chart'):
    try:
        # Retrieve live stock data
        data, meta_data = ts.get_intraday(selected_symbol, interval='5min', outputsize='compact')

        if not data.empty:
            st.write(f'Live Candlestick Chart for {selected_symbol}')

            # Create a candlestick chart using Plotly
            figure = go.Figure(data=[go.Candlestick(x=data.index,
                                                    open=data['1. open'],
                                                    high=data['2. high'],
                                                    low=data['3. low'],
                                                    close=data['4. close'])])

            st.plotly_chart(figure)
        else:
            st.write(f'No live data available for {selected_symbol}.')
    except Exception as e:
        st.error(f'Error: {str(e)}')





#Model Development
# Calculate the 10-day average trading volume
data['10_Day_Avg_Volume'] = data['Volume'].rolling(window=10).mean()
# Create a Streamlit app
st.title('Stock Trading Volume Analysis')
# Display the stock data
st.write(f'Stock Data for {ticker} for the last {selected_time_frame}:')
st.write(data)
data = data[data['10_Day_Avg_Volume'].notna()]
# Display the 10-day average trading volume
st.line_chart(data[['10_Day_Avg_Volume']])




# Define a function to calculate RSI
def calculate_rsi(data, period=14):
    # Calculate daily price changes
    delta = data['Close'].diff()
    # Calculate gains (positive price changes) and losses (negative price changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    # Calculate average gains and losses over the specified period
    avg_gains = gains.rolling(window=period, min_periods=1).mean()
    avg_losses = losses.rolling(window=period, min_periods=1).mean()
    # Calculate the relative strength (RS)
    rs = avg_gains / avg_losses
    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi
# Streamlit app
st.subheader('Relative Strength Index (RSI) Calculator')
# Upload historical price data
uploaded_file = data
# Specify the RSI calculation period
period = st.number_input('RSI Calculation Period', min_value=1, max_value=100, value=14)
# Calculate RSI
rsi = calculate_rsi(data, period)
# Display RSI values
st.header(f'RSI ({period}-day)')
st.write(rsi)
# Create a plot for RSI
st.line_chart(rsi)
# Specify the lag period
lag_period = st.number_input('Lag Period', min_value=1, max_value=len(data) - 1, value=1)





# Dynamically find the date column (ignoring leading/trailing spaces)
date_column = next((col for col in data.columns if col.strip().lower() == 'date'), None)
if date_column is not None:
    # Convert the date column to datetime and set it as the index
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)

    # Calculate MACD
    data = calculate_macd(data)

    # Calculate Bollinger Bands
    data = calculate_bollinger_bands(data)

    # Plot the data
    st.header('Historical Price Data with MACD and Bollinger Bands')
    st.line_chart(data[['Close', 'SMA', 'Upper_Band', 'Lower_Band']])
    st.line_chart(data[['MACD', 'Signal_Line']])

    # Candlestick chart
    st.header('Candlestick Chart')
    fig, ax = plt.subplots(figsize=(10, 6))
    ohlc_data = data[['Close', 'Close', 'Close', 'Close']].copy()
    ohlc_data.columns = ['Open', 'High', 'Low', 'Close']
    candlestick_ohlc(ax, ohlc_data.values, width=0.6, colorup='g', colordown='r')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    st.pyplot(fig)
else:
    st.error(f"No date column found in the uploaded data. Please check the column names.")





# Create lagged features
for column in data.columns:
    if column != 'Date':  # Exclude the 'Date' column if present
        data[f'{column}_lag_{lag_period}'] = data[column].shift(lag_period)

# Display the DataFrame with lagged features
st.header('DataFrame with Lagged Features')
st.write(data)





# Extract day of the week and month of the year as features
data['DayOfWeek'] = data.index.dayofweek
data['Month'] = data.index.month

# Display the DataFrame with seasonality features
st.header('DataFrame with Seasonality Features')
st.write(data)




# Calculate Average True Range (ATR)
atr_period = st.number_input('ATR Period', min_value=1, max_value=100, value=14)
data['ATR'] = data['High'] - data['Low']
data['ATR'] = data['ATR'].rolling(atr_period).mean()

# Display the DataFrame with ATR
st.header('DataFrame with ATR (Average True Range)')
st.write(data)







market_index_symbol = st.text_input('Enter Market Index Symbol (e.g., ^GSPC for S&P 500):')
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA12'] = data['Close_Stock'].ewm(span=short_window, adjust=False).mean()
    data['EMA26'] = data['Close_Stock'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Stock'] = data['EMA12'] - data['EMA26']
    data['Signal_Line_Stock'] = data['MACD_Stock'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Market'] = data['Close_Market'].ewm(span=short_window, adjust=False).mean()
    data['Signal_Line_Market'] = data['MACD_Market'].ewm(span=signal_window, adjust=False).mean()
    return data



# Function to fetch data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    # Download data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date)
    return data



# Fetch data
stock_data = fetch_data(ticker, start_date, end_date)

# Display fetched data
print(stock_data)
# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    data['SMA_Stock'] = data['Close_Stock'].rolling(window=window).mean()
    data['Upper_Band_Stock'] = data['SMA_Stock'] + (num_std_dev * data['Close_Stock'].rolling(window=window).std())
    data['Lower_Band_Stock'] = data['SMA_Stock'] - (num_std_dev * data['Close_Stock'].rolling(window=window).std())
    data['SMA_Market'] = data['Close_Market'].rolling(window=window).mean()
    data['Upper_Band_Market'] = data['SMA_Market'] + (num_std_dev * data['Close_Market'].rolling(window=window).std())
    data['Lower_Band_Market'] = data['SMA_Market'] - (num_std_dev * data['Close_Market'].rolling(window=window).std())
    return data

if start_date is not None and end_date is not None:
    # Retrieve historical price data using yfinance for the stock
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    st.write(f'Historical Price Data for {ticker}:')
    st.write(stock_data)

    # Retrieve historical price data using yfinance for the market index
    market_data = yf.download(ticker, start=start_date, end=end_date)

    st.write(f'Historical Price Data for {market_index_symbol}:')
    st.write(market_data)

    # Merge stock data and market data based on date
    merged_data = pd.merge(stock_data, market_data, left_index=True, right_index=True, suffixes=('_Stock', '_Market'))

    # Calculate MACD
    merged_data = calculate_macd(merged_data)

    # Calculate Bollinger Bands
    merged_data = calculate_bollinger_bands(merged_data)

    # Plot the data
    st.header(f'Historical Price Data for {ticker} and {market_index_symbol} with MACD and Bollinger Bands')
    st.line_chart(merged_data[['MACD_Stock', 'Signal_Line_Stock', 'MACD_Market', 'Signal_Line_Market']])

# Candlestick chart
st.header(f'Candlestick Chart for {ticker}')
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare the data with the correct column names and index formatting
ohlc_data = merged_data[['Open_Stock', 'High_Stock', 'Low_Stock', 'Close_Stock']].copy()
ohlc_data.columns = ['Open', 'High', 'Low', 'Close']
ohlc_data.reset_index(inplace=True)  # Reset index to make 'Date' a column
ohlc_data['Date'] = ohlc_data['Date'].map(mdates.date2num)  # Convert 'Date' to the required format

# Create the candlestick chart
candlestick_ohlc(ax, ohlc_data.values, width=0.6, colorup='g', colordown='r')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

st.pyplot(fig)



if start_date is not None and end_date is not None:
    # Retrieve historical price data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    st.write(f'Historical Price Data for {ticker}:')
    st.write(stock_data)

    # Calculate correlation matrix
    correlation_matrix = stock_data.corr()

    # Display the correlation matrix
    st.header('Correlation Matrix')
    st.write(correlation_matrix)

    # Select features with a correlation threshold
    correlation_threshold = st.slider('Correlation Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Select highly correlated features
    high_corr_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                high_corr_features.add(colname)

    # Display selected features
    st.header('Selected Features with High Correlation')
    st.write(high_corr_features)

    # Create a new DataFrame with selected features
    selected_data = stock_data[list(high_corr_features)]

    # Display the selected data
    st.header('Selected Data with High Correlation Features')
    st.write(selected_data)




# Normilization and Scaling


if start_date is not None and end_date is not None:
    # Retrieve historical price data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    st.write(f'Historical Price Data for {ticker}:')
    st.write(stock_data)

    # Select numerical columns for normalization
    numerical_columns = stock_data.select_dtypes(include=[float, int]).columns

    # Choose a scaling method (MinMaxScaler, StandardScaler, etc.)
    scaling_method = st.selectbox('Select Scaling Method:', ['MinMaxScaler', 'StandardScaler'])

    # Apply scaling
    if scaling_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:  # StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    scaled_data = stock_data.copy()
    scaled_data[numerical_columns] = scaler.fit_transform(stock_data[numerical_columns])

    # Display the scaled data
    st.header(f'Scaled Data using {scaling_method}')
    st.write(scaled_data)




# Train Test split
# Specify the split ratios
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

if start_date is not None and end_date is not None:
    # Retrieve historical price data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    st.write(f'Historical Price Data for {ticker}:')
    st.write(stock_data)

    # Calculate split points based on the ratios
    total_samples = len(stock_data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * validation_ratio)

    from sklearn.model_selection import train_test_split

    # Define the split ratios
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15

    # Calculate the sizes of the splits
    train_size = int(train_ratio * len(stock_data))
    validation_size = int(validation_ratio * len(stock_data))
    test_size = len(stock_data) - train_size - validation_size

    # Perform the split
    train_data, temp_data = train_test_split(stock_data, train_size=train_size, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=test_size, random_state=42)

    # Display the split datasets
    st.subheader('Training Data:')
    st.write(train_data)
    st.subheader('Validation Data:')
    st.write(validation_data)
    st.subheader('Test Data:')
    st.write(test_data)





























# Define the main function
def main():
    st.subheader(f"Real-Time Finance Data Analysis{ticker}")

    # Fetch historical data automatically upon entering the ticker symbol
    historical_data = fetch_historical_data(ticker)
    if historical_data is not None:
        display_analysis(ticker, historical_data)

# Function to fetch historical data for past several quarters
def fetch_historical_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period="1y")
        return historical_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to display analysis
def display_analysis(ticker, historical_data):


        # Plot line graph for closing prices
        fig, ax = plt.subplots()
        ax.plot(historical_data.index, historical_data['Close'], marker='o', linestyle='-')
        ax.set_title('Closing Prices Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        st.pyplot(fig)

        # Plot histogram for closing prices
        fig, ax = plt.subplots()
        ax.hist(historical_data['Close'], bins=20, color='skyblue', edgecolor='black')
        ax.set_title('Histogram of Closing Prices')
        ax.set_xlabel('Close Price')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# Run the main function
if __name__ == "__main__":
    main()














pricing_data,  news = st.tabs(["Pricing Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    data2=data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    st.write(data2)
    anual_return = data2['% Change'].mean()*252*100
    st.write("Annual Return Is ", anual_return,"%")
    stdev = np.std(data2["% Change"])*np.sqrt(252)
    st.write("Standard Devation is ", stdev*100,"%")
    st.write("Risk Adj. Retirn is ", anual_return/(stdev*100))



from stocknews import StockNews
with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {1 + 1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')



