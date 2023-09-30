import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import base64
import plotly.express as px
from datetime import datetime

stock_tickers = {
    "Tesla": "TSLA", "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", 
    "Facebook": "FB", "Amazon": "AMZN", "Netflix": "NFLX", "Alphabet": "GOOG", 
    "Nvidia": "NVDA", "Paypal": "PYPL", "Adobe": "ADBE", "Intel": "INTC", 
    "Cisco": "CSCO", "Comcast": "CMCSA", "Pepsi": "PEP", "Costco": "COST", 
    "Starbucks": "SBUX", "Walmart": "WMT", "Disney": "DIS", "Visa": "V", 
    "Mastercard": "MA", "Boeing": "BA", "IBM": "IBM", "McDonalds": "MCD", 
    "Nike": "NKE", "Exxon": "XOM", "Chevron": "CVX", "Verizon": "VZ", 
    "AT&T": "T", "Home Depot": "HD", "Salesforce": "CRM", "Oracle": "ORCL", 
    "Qualcomm": "QCOM", "AMD": "AMD"
}

st.title("Stock Comparison App")

# Sidebar for stock selection and date range specification
st.sidebar.header("Select Stocks and Date Ranges")

stock1 = st.sidebar.selectbox("Select Stock 1", list(stock_tickers.keys()), key="stock1_selectbox")
stock2 = st.sidebar.selectbox("Select Stock 2", list(stock_tickers.keys()), key="stock2_selectbox")

start_date_stock1 = st.sidebar.date_input(f"Start date for {stock1}", datetime(2010, 1, 1), key=f"start_date_{stock1}")
end_date_stock1 = st.sidebar.date_input(f"End date for {stock1}", datetime(2023, 1, 1), key=f"end_date_{stock1}")
start_date_stock2 = st.sidebar.date_input(f"Start date for {stock2}", datetime(2010, 1, 1), key=f"start_date_{stock2}")
end_date_stock2 = st.sidebar.date_input(f"End date for {stock2}", datetime(2023, 1, 1), key=f"end_date_{stock2}")

# Button to trigger the comparison
if st.sidebar.button("Compare"):
    # Load data for both stocks
    data_stock1 = yf.download(stock_tickers[stock1], start=start_date_stock1, end=end_date_stock1)
    data_stock2 = yf.download(stock_tickers[stock2], start=start_date_stock2, end=end_date_stock2)

    # Plotting the data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_stock1.index, y=data_stock1['Close'], mode='lines', name=f'{stock1}'))
    fig.add_trace(go.Scatter(x=data_stock2.index, y=data_stock2['Close'], mode='lines', name=f'{stock2}'))

    fig.update_layout(title=f"Comparison of {stock1} and {stock2}", xaxis_title="Date", yaxis_title="Closing Price")
    
    st.plotly_chart(fig)

st.sidebar.markdown("----")
st.sidebar.markdown("© 2023 Stock Comparison App")