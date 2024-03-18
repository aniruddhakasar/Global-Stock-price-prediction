import streamlit as st
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# Define the NSE website URL and headers
url = "https://www.nseindia.com/live_market/dynaContent/live_watch/get_quote/GetQuote.jsp?symbol=NIFTY+50"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299"
}

# Scrape the stock data from the NSE website
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
price_data = soup.find("span", {"id": "last_price"}).text
previous_close = soup.find("td", {"data-head": "Previous Close"}).find_next("td").text

# Create the data for the graph
date_labels = list(range(1, 61))
data = [
    go.Scatter(
        x=date_labels,
        y=[previous_close] + [int(price_data)] + [int(price_data)] * (59),
        mode="lines+markers",
        name="NIFTY 50"
    )
]

# Display the scraped data and the graph
st.write(f"NIFTY 50 (^NSEI)")
st.write(f"NSE - NSE Real Time Price. Currency in INR")
st.write(f"{price_data} +{previous_close - int(price_data)} ({previous_close - int(price_data)}%) As of now. Market open .")
st.write("Summary Chart Previous Close Open Volume ✩ Follow Conversations Historical Data")
st.write(f"Previous Close: {previous_close}")
st.write(f"Day's Range: {min(int(price_data), previous_close)} - {max(int(price_data), previous_close)}")
st.write(f"52 Week Range: 16,828.35-{max(int(price_data), previous_close)}")
st.write(f"Avg. Volume: 317,815")
st.plotly_chart(go.Figure(data=data), use_container_width=True)