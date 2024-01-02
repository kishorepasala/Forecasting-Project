#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA from statsmodels
import warnings
warnings.filterwarnings("ignore") 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


# In[2]:


st.markdown('''
<style>
.stApp {
    
    background-color:#8DC8ED;
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#000000;\
    border-style: false;\
    border-width: 2px;\
    color:Black;\
    font-size:15px;\
    font-family: Source Sans Pro;\
    background-color:#ffb3b3;\
    text-align:center;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}
.sidebar {
    background-color: black;
}
.st-b7 {
    color: #8DC8ED;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)

pickle_in = open(r"/Users/kishore/Desktop/arima_model.pkl", "rb")
arima_model = pickle.load(pickle_in)
brent_data_30 = pd.read_csv(r"/Users/kishore/Desktop/dataset.csv", header = None)


# In[3]:


brent_data_30


# In[4]:


st.title("Forecast Brent Crude oil Price")
st.sidebar.subheader("Select the number of days to Forecast from 2023-11-20")
days = st.sidebar.number_input('Days', min_value=1, step=1)


# In[5]:


# Create future dates
future = pd.date_range(start='2023-11-20', periods=days, tz=None, freq='D')
future_df = pd.DataFrame(index=future)

# Initialize the last 7 days data
data = brent_data_30[2].values


# In[6]:


future_df


# In[7]:


# Remove the first element
data_without_first = data[1:]

data = data_without_first


# In[21]:


data = data.astype(float)


# In[22]:


# Forecast for the selected number of days
forecast_values = []
for i in range(0, days):
    # Fit ARIMA model to the existing data
    arima_model = ARIMA(data, order=(2, 1, 1))
 # Replace p, d, and q with your model's order
    arima_fit = arima_model.fit()
    
    # Forecast one step ahead
    forecast = arima_fit.forecast()
    forecast_values.append(forecast[0])
    
    # Update the data for the next iteration
    data = np.append(data, forecast[0])

# Update the future_df with the forecasted data
future_df['oil price'] = forecast_values


# In[23]:


forecast_values


# In[24]:


future_df['oil price']


# In[26]:


# Display the forecast and data
st.sidebar.write(f"oil price for {days}th day")
st.sidebar.write(future_df[-1:])
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"oil price Forecasted for {days} days" )
    st.write(future_df)
with col2:
    st.subheader('Forecasted Graph')
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 3))
    ax.plot(future_df.index, future_df['oil price'], label='Forecast', color="orange")
    ax.tick_params(axis='x', labelrotation=100)
    plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True)
    plt.ylabel('oil price', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    st.pyplot(fig)

