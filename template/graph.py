import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st


# Data Frame
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
st.dataframe(df)

# Table
st.table(df)

# Line Chart
alt.Chart(df).mark_line().encode(alt.X("A", title="A"), alt.Y("B", title="B")).interactive()

# Map
alt.Chart(df).mark_circle().encode(alt.X("A", title="A"), alt.Y("B", title="B")).interactive()

# Slider
st.slider("Slider", 0, 10)

# Button
st.button("Click me!")

# Select Box
st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])

# Checkbox
st.checkbox("Check me!")

# File Upload
st.file_uploader("Upload a file")

# Radio Buttons
st.radio("Choose an option", ["Option 1", "Option 2", "Option 3"])

# Toggle
st.checkbox("Toggle me!")

# Alert
st.warning("This is an alert!")

# Confirm
st.button("Confirm me!")

# Cache
@st.cache
def expensive_computation(x):
    return np.sqrt(x)

st.write("Cached result: {}".format(expensive_computation(16)))

# Session State
if 'count' not in st.session_state:
    st.session_state.count = 0

def update_count():
    st.session_state.count += 1
    st.write("Count: {}".format(st.session_state.count))

st.button("Update Count", update_count)
