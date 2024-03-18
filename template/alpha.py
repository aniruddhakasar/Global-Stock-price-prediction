import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer



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

from stocknews import StockNews

def fetch_news(ticker, num_articles=10):
    sn = StockNews(ticker, save_news=False)
    news_df = sn.read_rss(num_articles=num_articles)
    return news_df
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from stocknews import StockNews

# Function to fetch historical stock data
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start="2022-01-01", end="2022-12-31")
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

# Function to fetch news for the given ticker
def fetch_news(ticker, num_articles=10):
    try:
        sn = StockNews(ticker, save_news=False)
        news_df = sn.read_rss(num_articles=num_articles)
        return news_df
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return None

# Main function
# Function to fetch historical stock data
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start="2022-01-01", end="2022-12-31")
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

# Function to fetch news for the given ticker
def fetch_news(ticker, num_articles=10):
    try:
        sn = StockNews(ticker, save_news=False)
        news_df = sn.read_rss()
        return news_df.head(num_articles)
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return None

# Main function
def main():
    st.title("Stock Price Prediction")

    # Sidebar for user input
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")

    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    if stock_data is not None:
        st.write("Stock Data:")
        st.write(stock_data.head())

    # Fetch news data
    num_articles = st.sidebar.number_input("Number of News Articles", value=10, min_value=1, max_value=50)
    news_df = fetch_news(ticker, num_articles=num_articles)
    if news_df is not None:
        st.write("News Data:")
        st.write(news_df)

if __name__ == "__main__":
    main()

# Function to train a machine learning model
def train_model(features, target):
    if features is None or target is None or len(features) == 0 or len(target) == 0:
        st.error("Error: Dataset is empty. Unable to train model.")
        return None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        return None, None, None


# Function to evaluate model accuracy
def evaluate_model(model, X_test, y_test):
    if model is None or X_test is None or y_test is None:
        st.error("Error: Unable to evaluate model accuracy.")
        return None

    try:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
        return None


# Function to predict next 10 days
def predict_next_10_days(model, recent_data):
    if model is None or recent_data is None:
        st.error("Error: Unable to make predictions.")
        return None

    try:
        predictions = model.predict(recent_data)
        return predictions
    except Exception as e:
        st.error(f"Error predicting next 10 days: {str(e)}")
        return None


# Main function
def main():
    st.title("Stock Price Prediction")

    # Sidebar for user input
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")

    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    if stock_data is not None:
        st.write("Stock Data:")
        st.write(stock_data.head())

    # Fetch news data
    num_articles = st.sidebar.number_input("Number of News Articles", value=10, min_value=1, max_value=50)
    news_df = fetch_news(ticker, num_articles=num_articles)
    if news_df is not None:
        st.write("News Data:")
        st.write(news_df)

if __name__ == "__main__":
    main()

# Main function
def main():
    st.title("Stock Price Prediction")


    # Fetch data and preprocess
    stock_data, news_df = fetch_stock_data_and_news(ticker)
    features, target = preprocess_data(stock_data, news_df)

    # Train model
    model, X_test, y_test = train_model(features, target)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    if accuracy is not None:
        st.write(f"Model Accuracy: {accuracy}")

    # Display predictions for the next 10 days
    st.subheader("Predictions for the Next 10 Days")
    recent_data = features.tail(1)
    next_10_days_dates = pd.date_range(start=recent_data.index[0], periods=10, freq='D')
    next_10_days_predictions = predict_next_10_days(model, recent_data)

    if next_10_days_predictions is not None:
        prediction_df = pd.DataFrame({
            'Date': next_10_days_dates,
            'Predicted Price Increase': next_10_days_predictions
        })

        st.write(prediction_df)


if __name__ == "__main__":
    main()
