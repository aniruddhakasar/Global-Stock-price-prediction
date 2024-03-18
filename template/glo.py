import pandas as pd
import streamlit as st
import numpy as np
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
    # Candlestick chart
    st.header(f'Candlestick Chart for {ticker}')
    fig, ax = plt.subplots(figsize=(10, 6))
    # Prepare the data with the correct column names and index formatting
    ohlc_data = merged_data[['Open_Stock', 'High_Stock', 'Low_Stock', 'Close_Stock']].copy()
    ohlc_data.columns = ['Open', 'High', 'Low', 'Close']
    ohlc_data.index = mdates.date2num(ohlc_data.index)
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

    # Perform the split
    train_data, temp_data = train_test_split(stock_data, train_size=train_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)

    # Display the split datasets
    st.header('Split Datasets')
    st.subheader('Training Data:')
    st.write(train_data)
    st.subheader('Validation Data:')
    st.write(val_data)
    st.subheader('Test Data:')
    st.write(test_data)

    # Load and preprocess the data
    if start_date is not None and end_date is not None:
        data = yf.download(ticker, start=start_date, end=end_date)
        st.write(f'Historical Price Data for {ticker}:')
        st.write(data)

        # Data Preprocessing
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

        # Define the look-back period and sequence length
        look_back = 10
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            y.append(scaled_data[i + look_back])

        X, y = np.array(X), np.array(y)

        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

        # Build the LSTM model
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=2)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write('Model Evaluation:')
        st.write(f'Mean Squared Error (MSE): {mse}')
        st.write(f'Mean Absolute Error (MAE): {mae}')
        st.write(f'R-squared (R2): {r2}')

        # Visualize the model's performance
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='True Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='red')
        plt.legend()
        plt.title('Stock Price Prediction')
        st.pyplot(plt)

        st.write('Training History:')
        st.line_chart(pd.DataFrame(history.history))

        # Save the model
        model.save('stock_price_prediction_model.h5')











































# Analysing And extracting data from the yohoo finance
Pricing_data, Fundamental_data, News, Quarter_Reports = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News", "Quarter_Reports"])
with Pricing_data:
    st.header("Price Movement")
    data2 = data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean() * 252 * 100
    st.write('Annual Return is :- ', annual_return, '%')
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    st.write('Standard Deiation is :- ', stdev * 100, '%')
    st.write('Risk Adj. Return is :- ', annual_return / (stdev * 100))

from alpha_vantage.fundamentaldata import FundamentalData
with Fundamental_data:
    key = 'KIEE5XOBEMJXB5LG'
    fd = FundamentalData(key, output_format="pandas")
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statment')
    income_statment = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statment.T[2:]
    is1.columns = list(income_statment.T.iloc[0])
    st.write(is1)
    st.subheader('cash Flow Statment')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)


from stocknews import StockNews
with News:
    st.subheader(f'News of {ticker}')
    sn = StockNews(ticker, save_news = False)
    df_news = sn.read_rss()
    for i in range(20):
        st.subheader(f'News {i+1}')
        st.write(df_news['Pubblished'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summery'][i])
        title_sentiments = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiments}')
        news_sentiments = df_news('sentiment_summery')[i]
        st.write(f'News Sentiment {news_sentiments}')
