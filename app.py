import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import streamlit as st
import plotly.graph_objects as go
import base64
import plotly.express as px
from datetime import datetime

# Convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Path to your local image
image_path = "images/stock_image.jpeg"
image_base64 = get_image_base64(image_path)

# Set the background using CSS
background_css = f"""
<style>
    body {{
        background-image: url("data:image/jpg;base64,{image_base64}");
        background-size: cover;
    }}
</style>
"""


st.set_page_config(page_title='Stock Price Analysis',  layout='wide', page_icon=':rocket:')

#this is the header
 

t1, t2 = st.columns((0.07,1)) 

t1.image('images/stock_image.jpeg', width = 80)
t2.title("Stock Price Analysis and Prediction Using LSTM")
t2.markdown(" **phone:** 01706 896161 **| website:** https://kowshik24.github.io/kowshik.github.io/ **| email:** kowshikcseruet1998@gmail.com")

# Add a dictonary of stock tickers and their company names and make a drop down menu to select the stock to predict

stock_tickers = {"Tesla": "TSLA", "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Facebook": "FB", "Amazon": "AMZN", "Netflix": "NFLX", "Alphabet": "GOOG", "Nvidia": "NVDA", "Paypal": "PYPL", "Adobe": "ADBE", "Intel": "INTC", "Cisco": "CSCO", "Comcast": "CMCSA", "Pepsi": "PEP", "Netflix": "NFLX", "Costco": "COST", "Starbucks": "SBUX", "Adobe": "ADBE", "Netflix": "NFLX", "Tesla": "TSLA", "Walmart": "WMT", "Disney": "DIS", "Visa": "V", "Mastercard": "MA", "Boeing": "BA", "IBM": "IBM", "McDonalds": "MCD", "Nike": "NKE", "Exxon": "XOM", "Chevron": "CVX", "Verizon": "VZ", "AT&T": "T", "Home Depot": "HD", "Salesforce": "CRM", "Oracle": "ORCL", "Qualcomm": "QCOM", "AMD": "AMD", "IBM": "IBM", "Cisco": "CSCO", "Intel": "INTC", "Nvidia": "NVDA", "Paypal": "PYPL", "Adobe": "ADBE", "Netflix": "NFLX", "Tesla": "TSLA", "Walmart": "WMT", "Disney": "DIS", "Visa": "V", "Mastercard": "MA", "Boeing": "BA", "IBM": "IBM", "McDonalds": "MCD", "Nike": "NKE", "Exxon": "XOM", "Chevron": "CVX", "Verizon": "VZ", "AT&T": "T", "Home Depot": "HD", "Salesforce": "CRM", "Oracle": "ORCL", "Qualcomm": "QCOM", "AMD": "AMD", "IBM": "IBM", "Cisco": "CSCO", "Intel": "INTC", "Nvidia": "NVDA", "Paypal": "PYPL", "Adobe": "AD"}
st.sidebar.title("Select a Stock")
# Custom CSS to change the sidebar color
sidebar_css = """
<style>
    div[data-testid="stSidebar"] > div:first-child {
        width: 350px;  # Adjust the width as needed
        background-color: #FF6969;
    }
</style>
"""

# User Input
default_index = stock_tickers.keys().index("TSLA") if "TSLA" in stock_tickers.keys() else 0
#st.markdown(sidebar_css, unsafe_allow_html=True)
user_input = st.sidebar.selectbox("Select a Stock", list(stock_tickers.keys()), index=default_index , key="main_selectbox")
stock_name = user_input
user_input = stock_tickers[user_input]

# User input for start and end dates using calendar widget
start_date = st.sidebar.date_input("Select start date:", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("Select end date:", datetime(2023, 1, 1))
# End of User Input



# Enhanced title with larger font size and a different color
title = f"<h1 style='color: red; font-size: 25px; text-align: center; '>{stock_name}'s Stock Analysis and Prediction Using LSTM</h1>"
st.markdown(title, unsafe_allow_html=True)
# Describing the data
st.subheader(f'Data from {start_date} - {end_date}')
data = yf.download(user_input, start_date, end_date)
# Reset the index to add the date column
data = data.reset_index()
# Display data in a Plotly table
fig = go.Figure(data=[go.Table(
        header=dict(values=list(data.columns),
                    font=dict(size=12, color='white'),
                    fill_color='#264653',
                    line_color='rgba(255,255,255,0.2)',
                    align=['left', 'center'],
                    height=20),
        cells=dict(values=[data[k].tolist() for k in data.columns],
                   font=dict(size=12),
                   align=['left', 'center'],
                   line_color='rgba(255,255,255,0.2)',
                   height=20))])

fig.update_layout(title_text=f"Data for {stock_name}", title_font_color='#264653', title_x=0, margin=dict(l=0, r=10, b=10, t=30))

st.plotly_chart(fig, use_container_width=True)


st.markdown(f"<h2 style='text-align: center; color: #264653;'>Data Overview for {stock_name}</h2>", unsafe_allow_html=True)
# Get the description of the data
description = data.describe()

# Dictionary of columns and rows to highlight
highlight_dict = {
    "Open": ["mean", "min", "max", "std"],
    "High": ["mean", "min", "max", "std"],
    "Low": ["mean", "min", "max", "std"],
    "Close": ["mean", "min", "max", "std"],
    "Adj Close": ["mean", "min", "max", "std"]
}

# Colors for specific rows
color_dict = {
    "mean": "lightgreen",
    "min": "salmon",
    "max": "lightblue",
    "std": "lightyellow"
}

# Function to highlight specific columns and rows based on the dictionaries
def highlight_specific_cells(val, col_name, row_name):
    if col_name in highlight_dict and row_name in highlight_dict[col_name]:
        return f'background-color: {color_dict[row_name]}'
    return ''

styled_description = description.style.apply(lambda row: [highlight_specific_cells(val, col, row.name) for col, val in row.items()], axis=1)

# Display the styled table in Streamlit
st.table(styled_description)






### ............................................... ##
# Stock Price Over Time
g1, g2, g3 = st.columns((1.2,1.2,1))

fig1 = px.line(data, x='Date', y='Close', template='seaborn')
fig1.update_traces(line_color='#264653')
fig1.update_layout(title_text="Stock Price Over Time", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, height=400, width=700)
g1.plotly_chart(fig1, use_container_width=True)

# Volume of Stocks Traded Over Time
fig2 = px.bar(data, x='Date', y='Volume', template='seaborn')
fig2.update_traces(marker_color='#7A9E9F')
fig2.update_layout(title_text="Volume of Stocks Traded Over Time", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, height=400, width=700)
g2.plotly_chart(fig2, use_container_width=True)

# Moving Averages
short_window = 40
long_window = 100
data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
fig3 = px.line(data, x='Date', y='Close', template='seaborn')
fig3.add_scatter(x=data['Date'], y=data['Short_MA'], mode='lines', line=dict(color="red"), name=f'Short {short_window}D MA')
fig3.add_scatter(x=data['Date'], y=data['Long_MA'], mode='lines', line=dict(color="blue"), name=f'Long {long_window}D MA')
fig3.update_layout(title_text="Stock Price with Moving Averages", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="right", x=0.99), height=400, width=700)
g3.plotly_chart(fig3, use_container_width=True)






## ............................................... ##
# Daily Returns
g4, g5, g6 = st.columns((1,1,1))
data['Daily_Returns'] = data['Close'].pct_change()
fig4 = px.line(data, x='Date', y='Daily_Returns', template='seaborn')
fig4.update_traces(line_color='#E76F51')
fig4.update_layout(title_text="Daily Returns", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g4.plotly_chart(fig4, use_container_width=True)

# Cumulative Returns
data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod()
fig5 = px.line(data, x='Date', y='Cumulative_Returns', template='seaborn')
fig5.update_traces(line_color='#2A9D8F')
fig5.update_layout(title_text="Cumulative Returns", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g5.plotly_chart(fig5, use_container_width=True)
# Stock Price Distribution
fig6 = px.histogram(data, x='Close', template='seaborn', nbins=50)
fig6.update_traces(marker_color='#F4A261')
fig6.update_layout(title_text="Stock Price Distribution", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g6.plotly_chart(fig6, use_container_width=True)

## ............................................... ##

# Bollinger Bands
g7, g8, g9 = st.columns((1,1,1))
rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()
data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
fig7 = px.line(data, x='Date', y='Close', template='seaborn')
fig7.add_scatter(x=data['Date'], y=data['Bollinger_Upper'], mode='lines', line=dict(color="green"), name='Upper Bollinger Band')
fig7.add_scatter(x=data['Date'], y=data['Bollinger_Lower'], mode='lines', line=dict(color="red"), name='Lower Bollinger Band')
fig7.update_layout(title_text="Bollinger Bands", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g7.plotly_chart(fig7, use_container_width=True)

# Stock Price vs. Volume
fig8 = px.line(data, x='Date', y='Close', template='seaborn')
fig8.add_bar(x=data['Date'], y=data['Volume'], name='Volume')
fig8.update_layout(title_text="Stock Price vs. Volume", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g8.plotly_chart(fig8, use_container_width=True)

# MACD
data['12D_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
data['26D_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['12D_EMA'] - data['26D_EMA']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
fig9 = px.line(data, x='Date', y='MACD', template='seaborn', title="MACD")
fig9.add_scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', line=dict(color="orange"), name='Signal Line')
fig9.update_layout(title_text="MACD", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g9.plotly_chart(fig9, use_container_width=True)


### ............................................... ##

# Relative Strength Index (RSI)
g10, g11, g12 = st.columns((1,1,1))
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
fig10 = px.line(data, x='Date', y='RSI', template='seaborn')
fig10.update_layout(title_text="Relative Strength Index (RSI)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g10.plotly_chart(fig10, use_container_width=True)

# Candlestick Chart
fig11 = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
fig11.update_layout(title_text="Candlestick Chart", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
g11.plotly_chart(fig11, use_container_width=True)

# Correlation Matrix
corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
fig12 = px.imshow(corr_matrix, template='seaborn')
fig12.update_layout(title_text="Correlation Matrix", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
g12.plotly_chart(fig12, use_container_width=True)

### ............................................... ##
# Price Rate of Change (ROC)
g13, g14, g15 = st.columns((1,1,1))
n = 12
data['ROC'] = ((data['Close'] - data['Close'].shift(n)) / data['Close'].shift(n)) * 100
fig13 = px.line(data, x='Date', y='ROC', template='seaborn')
fig13.update_layout(title_text="Price Rate of Change (ROC)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g13.plotly_chart(fig13, use_container_width=True)

# Stochastic Oscillator
low_min = data['Low'].rolling(window=14).min()
high_max = data['High'].rolling(window=14).max()
data['%K'] = (100 * (data['Close'] - low_min) / (high_max - low_min))
data['%D'] = data['%K'].rolling(window=3).mean()
fig14 = px.line(data, x='Date', y='%K', template='seaborn')
fig14.add_scatter(x=data['Date'], y=data['%D'], mode='lines', line=dict(color="orange"), name='%D (3-day SMA of %K)')
fig14.update_layout(title_text="Stochastic Oscillator", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g14.plotly_chart(fig14, use_container_width=True)

# Historical Volatility
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Historical_Volatility'] = data['Log_Return'].rolling(window=252).std() * np.sqrt(252)
fig15 = px.line(data, x='Date', y='Historical_Volatility', template='seaborn')
fig15.update_layout(title_text="Historical Volatility (252-day)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
g15.plotly_chart(fig15, use_container_width=True)

### ............................................... ##

# Visualizing the data and want to get the data when hovering over the graph
st.subheader('Closing Price vs Time Chart')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
fig1.layout.update(hovermode='x')
# Display the figure in Streamlit
st.plotly_chart(fig1,use_container_width=True)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data['Close'].rolling(100).mean()
fig2 = go.Figure()
# Add traces for 100MA and Closing Price
fig2.add_trace(go.Scatter(x=data.index, y=ma100, mode='lines', name='100MA'))
fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
fig2.layout.update(hovermode='x')
# Display the figure in Streamlit
st.plotly_chart(fig2,use_container_width=True)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = data['Close'].rolling(100).mean()
ma200 = data['Close'].rolling(200).mean()
fig3 = go.Figure()
# Add traces for 100MA and Closing Price
fig3.add_trace(go.Scatter(x=data.index, y=ma100, mode='lines', name='100MA'))
fig3.add_trace(go.Scatter(x=data.index, y=ma200, mode='lines', name='200MA'))
fig3.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price'))
fig3.layout.update(hovermode='x')
# Display the figure in Streamlit
st.plotly_chart(fig3,use_container_width=True)


# Splitting the data into training and testing data
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# load the model
model = load_model('best_model_tesla.h5')

# Testing the model
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Visualizing the results
st.subheader('Predictions vs Actual')
fig4 = go.Figure()
# Add traces for Actual and Predicted Price
fig4.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='Actual Price'))
fig4.add_trace(go.Scatter(x=data.index[-len(y_predicted):], y=y_predicted[:,0], mode='lines', name='Predicted Price'))
fig4.layout.update(hovermode='x')
# Display the figure in Streamlit
st.plotly_chart(fig4,use_container_width=True)





# Contact Form
with st.expander("Contact us"):
    with st.form(key='contact', clear_on_submit=True):
        
        email = st.text_input('Contact Email')
        st.text_area("Query",placeholder="Please fill in all the information or we may not be able to process your request")  
        
        submit_button = st.form_submit_button(label='Send Information')
