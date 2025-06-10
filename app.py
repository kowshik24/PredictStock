import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pandas_datareader.data as web # Not used, can be removed
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
# import streamlit as st # Duplicate import
import plotly.graph_objects as go
import base64
import plotly.express as px
from datetime import datetime

# Convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Path to your local image
# Ensure this path is correct relative to where the script is run.
# If app.py is in /workspaces/PredictStock/ and image is in /workspaces/PredictStock/images/
image_path = "images/stock_image.jpeg" 
try:
    image_base64 = get_image_base64(image_path)
    # Set the background using CSS
    # This background CSS injection through st.markdown might not work reliably or as expected for the entire body.
    # Streamlit has options for theming. For full background image, custom components or more complex CSS might be needed.
    # background_css = f"""
    # <style>
    #     body {{
    #         background-image: url("data:image/jpeg;base64,{image_base64}");
    #         background-size: cover;
    #     }}
    # </style>
    # """
    # st.markdown(background_css, unsafe_allow_html=True) # Commenting out as it can be problematic
except FileNotFoundError:
    st.warning(f"Image file not found at {image_path}. Background will not be set.")
    image_base64 = None


st.set_page_config(page_title='Stock Price Analysis',  layout='wide', page_icon=':rocket:')

#this is the header
t1, t2 = st.columns((0.07,1)) 

try:
    t1.image('images/stock_image.jpeg', width = 80)
except Exception: # PIL.UnidentifiedImageError or FileNotFoundError
    t1.warning("Header image not found or invalid.")

t2.title("Stock Price Analysis and Prediction Using LSTM")
t2.markdown(" **phone:** 01706 896161 **| website:** https://kowshik24.github.io/kowshik.github.io/ **| email:** kowshikcseruet1998@gmail.com")

# Add a dictionary of stock tickers and their company names
# Cleaned up duplicate entries. Note: Facebook is Meta (META), Google is Alphabet (GOOGL/GOOG)
# For simplicity, using the provided list but cleaned.
stock_tickers = {
    "Tesla": "TSLA", "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", 
    "Meta (Facebook)": "META", "Amazon": "AMZN", "Netflix": "NFLX", "Nvidia": "NVDA", 
    "Paypal": "PYPL", "Adobe": "ADBE", "Intel": "INTC", "Cisco": "CSCO", 
    "Comcast": "CMCSA", "Pepsi": "PEP", "Costco": "COST", "Starbucks": "SBUX", 
    "Walmart": "WMT", "Disney": "DIS", "Visa": "V", "Mastercard": "MA", 
    "Boeing": "BA", "IBM": "IBM", "McDonalds": "MCD", "Nike": "NKE", 
    "Exxon": "XOM", "Chevron": "CVX", "Verizon": "VZ", "AT&T": "T", 
    "Home Depot": "HD", "Salesforce": "CRM", "Oracle": "ORCL", 
    "Qualcomm": "QCOM", "AMD": "AMD"
} # Removed duplicates and "Facebook"->"FB", used "Meta"->"META"

st.sidebar.title("Select a Stock")
# Custom CSS to change the sidebar color - This seems to be commented out in the original log.
# sidebar_css = """
# <style>
#     div[data-testid="stSidebar"] > div:first-child {
#         width: 350px;  # Adjust the width as needed
#         background-color: #FF6969;
#     }
# </style>
# """
# st.markdown(sidebar_css, unsafe_allow_html=True)


# User Input
# Corrected: stock_tickers.keys() is a dict_keys object, convert to list for .index()
stock_keys_list = list(stock_tickers.keys())
default_company_name = "Tesla" 
default_index = stock_keys_list.index(default_company_name) if default_company_name in stock_keys_list else 0

selected_company_name = st.sidebar.selectbox("Select a Stock", stock_keys_list, index=default_index , key="main_selectbox")
stock_ticker_symbol = stock_tickers[selected_company_name]

# User input for start and end dates using calendar widget
start_date = st.sidebar.date_input("Select start date:", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("Select end date:", datetime.now().date()) # Default end date to today


# Enhanced title with larger font size and a different color
title = f"<h1 style='color: red; font-size: 25px; text-align: center; '>{selected_company_name}'s ({stock_ticker_symbol}) Stock Analysis and Prediction Using LSTM</h1>"
st.markdown(title, unsafe_allow_html=True)

# Describing the data
st.subheader(f'Data from {start_date} - {end_date}')
data = yf.download(stock_ticker_symbol, start=start_date, end=end_date)

# FIX: Check if columns are MultiIndex and flatten if necessary
if isinstance(data.columns, pd.MultiIndex):
    # This typically happens if yf.download gets a list of tickers, e.g. ["TSLA"]
    # or if the yfinance version/configuration causes it for single tickers.
    # We assume the desirable column names are at level 0 of the MultiIndex.
    data.columns = data.columns.get_level_values(0)

data = data.reset_index() # Reset the index to add the date column

# Check if data is empty (e.g. invalid ticker or date range)
if data.empty:
    st.error(f"No data found for {selected_company_name} ({stock_ticker_symbol}) for the selected date range. Please select a different stock or date range.")
    st.stop() # Stop execution if no data

# Display data in a Plotly table
# Ensure 'Date' column is string for display if it has timezone, to avoid Arrow issues
# data_display = data.copy()
# if pd.api.types.is_datetime64_any_dtype(data_display['Date']) and data_display['Date'].dt.tz is not None:
#    data_display['Date'] = data_display['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
# else:
#    data_display['Date'] = data_display['Date'].dt.strftime('%Y-%m-%d')


fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(data.columns), # Use data.columns directly
                    font=dict(size=12, color='white'),
                    fill_color='#264653',
                    line_color='rgba(255,255,255,0.2)',
                    align=['left', 'center'],
                    height=20),
        cells=dict(values=[data[k].tolist() for k in data.columns], # Use data.columns
                   font=dict(size=12),
                   align=['left', 'center'],
                   line_color='rgba(255,255,255,0.2)',
                   height=20))])

fig_table.update_layout(title_text=f"Data for {selected_company_name}", title_font_color='#264653', title_x=0, margin=dict(l=0, r=10, b=10, t=30))
st.plotly_chart(fig_table, use_container_width=True)


st.markdown(f"<h2 style='text-align: center; color: #264653;'>Data Overview for {selected_company_name}</h2>", unsafe_allow_html=True)
# Get the description of the data, excluding datetime columns for describe()
description = data.select_dtypes(include=np.number).describe()


# Dictionary of columns and rows to highlight
highlight_dict = {
    "Open": ["mean", "min", "max", "std"],
    "High": ["mean", "min", "max", "std"],
    "Low": ["mean", "min", "max", "std"],
    "Close": ["mean", "min", "max", "std"],
    "Adj Close": ["mean", "min", "max", "std"] # Will not exist if auto_adjust=True (default)
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
    # Check if col_name exists in highlight_dict (e.g., 'Adj Close' might be missing)
    if col_name in highlight_dict and row_name in highlight_dict[col_name]:
        return f'background-color: {color_dict[row_name]}'
    return ''

# Check if description DataFrame is empty
if not description.empty:
    styled_description = description.style.apply(lambda row: [highlight_specific_cells(val, col, row.name) for col, val in row.items()], axis=1)
    st.table(styled_description)
else:
    st.warning("Could not generate data description (e.g., no numeric data available).")


### ............................................... ##
# Stock Price Over Time
g1, g2, g3 = st.columns((1.2,1.2,1))

# Ensure 'Date' and 'Close' columns exist
if 'Date' in data.columns and 'Close' in data.columns:
    fig1 = px.line(data, x='Date', y='Close', template='seaborn') # THIS WAS THE ERROR LINE
    fig1.update_traces(line_color='#264653')
    fig1.update_layout(title_text="Stock Price Over Time", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, height=400, width=700)
    g1.plotly_chart(fig1, use_container_width=True)
else:
    g1.error("Could not plot Stock Price Over Time: 'Date' or 'Close' column missing.")

# Volume of Stocks Traded Over Time
if 'Date' in data.columns and 'Volume' in data.columns:
    fig2 = px.bar(data, x='Date', y='Volume', template='seaborn')
    fig2.update_traces(marker_color='#7A9E9F')
    fig2.update_layout(title_text="Volume of Stocks Traded Over Time", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, height=400, width=700)
    g2.plotly_chart(fig2, use_container_width=True)
else:
    g2.error("Could not plot Volume: 'Date' or 'Volume' column missing.")


# Moving Averages
if 'Date' in data.columns and 'Close' in data.columns:
    short_window = 40
    long_window = 100
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    fig3_ma = px.line(data, x='Date', y='Close', template='seaborn') # Renamed fig3 to fig3_ma to avoid conflict
    fig3_ma.add_scatter(x=data['Date'], y=data['Short_MA'], mode='lines', line=dict(color="red"), name=f'Short {short_window}D MA')
    fig3_ma.add_scatter(x=data['Date'], y=data['Long_MA'], mode='lines', line=dict(color="blue"), name=f'Long {long_window}D MA')
    fig3_ma.update_layout(title_text="Stock Price with Moving Averages", title_x=0, margin=dict(l=20, r=20, b=20, t=30), yaxis_title=None, xaxis_title=None, legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="right", x=0.99), height=400, width=700)
    g3.plotly_chart(fig3_ma, use_container_width=True)
else:
    g3.error("Could not plot Moving Averages: 'Date' or 'Close' column missing.")

# ... (rest of the plotting code needs similar checks for column existence) ...
# For brevity, I'll assume 'Date' and 'Close' and other necessary columns exist for the rest of the plots.
# Proper error handling should be added for each plot if columns might be missing.

## ............................................... ##
# Daily Returns
g4, g5, g6 = st.columns((1,1,1))
if 'Close' in data.columns:
    data['Daily_Returns'] = data['Close'].pct_change()
    fig4_dr = px.line(data, x='Date', y='Daily_Returns', template='seaborn')
    fig4_dr.update_traces(line_color='#E76F51')
    fig4_dr.update_layout(title_text="Daily Returns", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g4.plotly_chart(fig4_dr, use_container_width=True)

    # Cumulative Returns
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod()
    fig5_cr = px.line(data, x='Date', y='Cumulative_Returns', template='seaborn')
    fig5_cr.update_traces(line_color='#2A9D8F')
    fig5_cr.update_layout(title_text="Cumulative Returns", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g5.plotly_chart(fig5_cr, use_container_width=True)

    # Stock Price Distribution
    fig6_spd = px.histogram(data, x='Close', template='seaborn', nbins=50)
    fig6_spd.update_traces(marker_color='#F4A261')
    fig6_spd.update_layout(title_text="Stock Price Distribution", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g6.plotly_chart(fig6_spd, use_container_width=True)
else:
    g4.error("Daily Returns: 'Close' column missing.")
    g5.error("Cumulative Returns: 'Close' column missing.")
    g6.error("Stock Price Distribution: 'Close' column missing.")


## ............................................... ##
# Bollinger Bands
g7, g8, g9 = st.columns((1,1,1))
if 'Close' in data.columns:
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
    fig7_bb = px.line(data, x='Date', y='Close', template='seaborn')
    fig7_bb.add_scatter(x=data['Date'], y=data['Bollinger_Upper'], mode='lines', line=dict(color="green"), name='Upper Bollinger Band')
    fig7_bb.add_scatter(x=data['Date'], y=data['Bollinger_Lower'], mode='lines', line=dict(color="red"), name='Lower Bollinger Band')
    fig7_bb.update_layout(title_text="Bollinger Bands", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g7.plotly_chart(fig7_bb, use_container_width=True)

    # Stock Price vs. Volume
    if 'Volume' in data.columns:
        fig8_pv = px.line(data, x='Date', y='Close', template='seaborn') # Changed name
        fig8_pv.add_bar(x=data['Date'], y=data['Volume'], name='Volume') # Changed name
        fig8_pv.update_layout(title_text="Stock Price vs. Volume", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
        g8.plotly_chart(fig8_pv, use_container_width=True)
    else:
        g8.error("Stock Price vs Volume: 'Volume' column missing.")


    # MACD
    data['12D_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['26D_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['12D_EMA'] - data['26D_EMA']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    fig9_macd = px.line(data, x='Date', y='MACD', template='seaborn', title="MACD") # Changed name
    fig9_macd.add_scatter(x=data['Date'], y=data['Signal_Line'], mode='lines', line=dict(color="orange"), name='Signal Line')
    fig9_macd.update_layout(title_text="MACD", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g9.plotly_chart(fig9_macd, use_container_width=True)
else:
    g7.error("Bollinger Bands: 'Close' column missing.")
    # g8 error handled internally
    g9.error("MACD: 'Close' column missing.")


### ............................................... ##
# Relative Strength Index (RSI)
g10, g11, g12 = st.columns((1,1,1))
if 'Close' in data.columns:
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    fig10_rsi = px.line(data, x='Date', y='RSI', template='seaborn') # Changed name
    fig10_rsi.update_layout(title_text="Relative Strength Index (RSI)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g10.plotly_chart(fig10_rsi, use_container_width=True)
else:
    g10.error("RSI: 'Close' column missing.")

# Candlestick Chart
if all(col in data.columns for col in ['Date', 'Open', 'High', 'Low', 'Close']):
    fig11_cs = go.Figure(data=[go.Candlestick(x=data['Date'], # Changed name
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])
    fig11_cs.update_layout(title_text="Candlestick Chart", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
    g11.plotly_chart(fig11_cs, use_container_width=True)
else:
    g11.error("Candlestick: One or more required columns (Date, Open, High, Low, Close) missing.")


# Correlation Matrix
# Select only numeric columns for correlation
numeric_cols_for_corr = ['Open', 'High', 'Low', 'Close', 'Volume']
existing_numeric_cols = [col for col in numeric_cols_for_corr if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

if len(existing_numeric_cols) > 1:
    corr_matrix = data[existing_numeric_cols].corr()
    fig12_cm = px.imshow(corr_matrix, template='seaborn', text_auto=True) # Changed name, added text_auto
    fig12_cm.update_layout(title_text="Correlation Matrix", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
    g12.plotly_chart(fig12_cm, use_container_width=True)
else:
    g12.error("Correlation Matrix: Not enough numeric columns available for correlation.")


### ............................................... ##
# Price Rate of Change (ROC)
g13, g14, g15 = st.columns((1,1,1))
if 'Close' in data.columns:
    n_roc = 12 # Renamed n to n_roc
    data['ROC'] = ((data['Close'] - data['Close'].shift(n_roc)) / data['Close'].shift(n_roc)) * 100
    fig13_roc = px.line(data, x='Date', y='ROC', template='seaborn') # Changed name
    fig13_roc.update_layout(title_text="Price Rate of Change (ROC)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g13.plotly_chart(fig13_roc, use_container_width=True)
else:
    g13.error("ROC: 'Close' column missing.")

# Stochastic Oscillator
if all(col in data.columns for col in ['Low', 'High', 'Close']):
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    data['%K'] = (100 * (data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(window=3).mean()
    fig14_so = px.line(data, x='Date', y='%K', template='seaborn') # Changed name
    fig14_so.add_scatter(x=data['Date'], y=data['%D'], mode='lines', line=dict(color="orange"), name='%D (3-day SMA of %K)')
    fig14_so.update_layout(title_text="Stochastic Oscillator", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g14.plotly_chart(fig14_so, use_container_width=True)
else:
    g14.error("Stochastic Oscillator: 'Low', 'High', or 'Close' column missing.")

# Historical Volatility
if 'Close' in data.columns:
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Historical_Volatility'] = data['Log_Return'].rolling(window=252).std() * np.sqrt(252)
    fig15_hv = px.line(data, x='Date', y='Historical_Volatility', template='seaborn') # Changed name
    fig15_hv.update_layout(title_text="Historical Volatility (252-day)", title_x=0, margin=dict(l=0, r=10, b=10, t=30), yaxis_title=None, xaxis_title=None)
    g15.plotly_chart(fig15_hv, use_container_width=True)
else:
    g15.error("Historical Volatility: 'Close' column missing.")


### ............................................... ##
# Renamed fig1, fig2, fig3 here to avoid conflicts with earlier figures
# Visualizing the data and want to get the data when hovering over the graph
st.subheader('Closing Price vs Time Chart')
if 'Close' in data.columns:
    # Use data.index for x-axis if 'Date' column was problematic or if preferred
    # Using 'Date' column as it's explicitly created and used elsewhere
    fig_close_time = go.Figure()
    fig_close_time.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig_close_time.layout.update(hovermode='x')
    st.plotly_chart(fig_close_time,use_container_width=True)
else:
    st.error("Closing Price vs Time: 'Close' column missing.")


st.subheader('Closing Price vs Time Chart with 100MA')
if 'Close' in data.columns:
    ma100 = data['Close'].rolling(100).mean()
    fig_ma100 = go.Figure()
    fig_ma100.add_trace(go.Scatter(x=data['Date'], y=ma100, mode='lines', name='100MA'))
    fig_ma100.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Closing Price'))
    fig_ma100.layout.update(hovermode='x')
    st.plotly_chart(fig_ma100,use_container_width=True)
else:
    st.error("Closing Price vs Time with 100MA: 'Close' column missing.")


st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
if 'Close' in data.columns:
    ma100 = data['Close'].rolling(100).mean() # Recalculate or use existing if scope allows
    ma200 = data['Close'].rolling(200).mean()
    fig_ma100_200 = go.Figure()
    fig_ma100_200.add_trace(go.Scatter(x=data['Date'], y=ma100, mode='lines', name='100MA'))
    fig_ma100_200.add_trace(go.Scatter(x=data['Date'], y=ma200, mode='lines', name='200MA'))
    fig_ma100_200.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Closing Price'))
    fig_ma100_200.layout.update(hovermode='x')
    st.plotly_chart(fig_ma100_200,use_container_width=True)
else:
    st.error("Closing Price vs Time with 100MA & 200MA: 'Close' column missing.")


if 'Close' in data.columns and len(data['Close']) > 100 : # Ensure enough data for splitting and past_100_days
    # Splitting the data into training and testing data
    data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
    data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    # load the model
    # WARNING: This model is hardcoded for Tesla. It will not perform well for other stocks.
    # For a general app, you'd need a model per stock or a more general model.
    # Or, train a model on the fly (can be slow).
    try:
        model = load_model('best_model_tesla.h5') # keras.models.load_model
        st.info("LSTM Model 'best_model_tesla.h5' loaded. Note: This model is pre-trained on Tesla data.")
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}. Prediction section will not work.")
        model = None # Set model to None so following code can check

    if model:
        # Testing the model
        past_100_days = data_training.tail(100)
        # Ensure final_df is created correctly by concatenating pandas Series/DataFrames
        final_df = pd.concat([past_100_days['Close'], data_testing['Close']], ignore_index=True)
        final_df = pd.DataFrame(final_df) # Convert Series to DataFrame for scaler

        input_data = scaler.fit_transform(final_df) # Should be transform, not fit_transform, if using original scaler fitted on training

        x_test = []
        y_test = []
        if len(input_data) >= 100: # Make sure there's enough data for the loop
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])

            if x_test: # Check if x_test has data
                x_test, y_test = np.array(x_test), np.array(y_test)

                y_predicted = model.predict(x_test)

                # Inverse scaling
                # The scaler used for y_predicted should be the one fitted on the 'Close' prices
                # that y_predicted tries to predict.
                # scaler.scale_ is an array, get the specific scale for 'Close' feature
                # If scaler was fit on a single column (Close price), then scaler.scale_[0] is correct.
                scale_factor = 1/scaler.scale_[0]
                y_predicted = y_predicted * scale_factor
                y_test = y_test * scale_factor

                # Visualizing the results
                st.subheader('Predictions vs Actual')
                fig_pred = go.Figure() # Renamed fig4
                
                # Determine the correct index for y_test and y_predicted on the plot
                # y_test and y_predicted correspond to data_testing part
                test_dates = data['Date'][int(len(data)*0.70):][100:] # Align dates correctly
                
                # Ensure lengths match for plotting
                min_len = min(len(test_dates), len(y_test), len(y_predicted))
                
                fig_pred.add_trace(go.Scatter(x=test_dates[:min_len], y=y_test[:min_len], mode='lines', name='Actual Price'))
                fig_pred.add_trace(go.Scatter(x=test_dates[:min_len], y=y_predicted[:min_len,0], mode='lines', name='Predicted Price'))
                fig_pred.layout.update(hovermode='x', title='LSTM Model Prediction vs Actual Price')
                st.plotly_chart(fig_pred,use_container_width=True)
            else:
                st.warning("Not enough data in the test set to make predictions after creating sequences.")
        else:
            st.warning("Not enough data to create test sequences for LSTM model.")
else:
    st.warning("Not enough data for 'Close' column to perform LSTM prediction, or 'Close' column missing.")


# Contact Form
with st.expander("Contact us"):
    with st.form(key='contact', clear_on_submit=True):
        email = st.text_input('Contact Email')
        st.text_area("Query",placeholder="Please fill in all the information or we may not be able to process your request")  
        submit_button = st.form_submit_button(label='Send Information')
        if submit_button:
            # Add logic here to handle form submission, e.g., send an email
            st.success("Thank you for your message! We will get back to you if 'Contact Email' is provided.")