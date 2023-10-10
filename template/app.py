import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras.models import load_model
import streamlit as st
import datetime

# Define the date range
start_date = datetime.datetime(2021, 1, 1)
end_date = datetime.datetime(2021, 12, 31)



st.title("Global Stock Price Prediction Using Deep Learning")


user_input = st.text_input("Enter Stock Ticker")
df = pdr.DataReader(user_input, 'yahoo', start_date, end_date)


#Describing data 
st.subheader("Data From 2001 - 2024")
st.write(df.describe())

#visulation
st.subheader("Closing Price vs Time Chart")
ma100 = df.close.rolling(100).mean()
fig=plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)