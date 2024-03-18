import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras.models import load_model
import streamlit as st
import yfinance as yf



# Define the date range
start_date = '2010-01-01'
end_date = '2021-12-31'



st.title("Global Stock Price Prediction Using Deep Learning")


user_input = input("Enter Stock Ticker")
data = yf.download(user_input,"Yahoo", start_date,end_date)


#Describing data 
st.subheader("Data From 2001 - 2024")
st.write(data.describe())

#visulation
#st.subheader("Closing Price vs Time Chart")
#ma100 = df.close.rolling(100).mean()
#fig=plt.figure(figsize=(12, 6))
#plt.plot(ma100)
#plt.plot(df.close)
#st.pyplot(fig)