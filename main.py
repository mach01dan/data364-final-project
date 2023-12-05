## Set up instructions: Install the required packages using the following command:
## pip install streamlit yfinance prophet requests plotly pandas ta
## Run the app using the following command:
## streamlit run main.py

import streamlit as st
from datetime import date, time, datetime
import yfinance as yf
from prophet import Prophet
import requests
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import ta
import os, contextlib
from os.path import isfile, join

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks_dir = "stocks"
dataframes = []
data = None  # Initialize data in the global scope

realtime_api_key = 'IDKN0O488FIDEPVR'
news_api_key = '60d16a8cc922453896de9495f9b99b1d'

# Set threshold values for the indicators
rsi_buy_threshold = 45  # Lower RSI values suggest buying opportunities
rsi_sell_threshold = 55  # Higher RSI values suggest selling opportunities
macd_buy_threshold = 0  # Higher MACD values relative to the signal line suggest buying opportunities
macd_sell_threshold = 0  # Lower MACD values relative to the signal line suggest selling opportunities


##### FUNCTIONS SECTION BEGINS #####

def download_stock_data():
    offset = 0
    limit = 3000
    period = '5y'  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

    if not os.path.exists(stocks_dir):
        os.makedirs(stocks_dir)
        print(f"The folder '{stocks_dir}' has been created.")
    else:
        print(f"The folder '{stocks_dir}' already exists.")

    symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    limit = limit if limit else len(symbols)
    end = min(offset + limit, len(symbols))
    is_valid = [False] * len(symbols)
    st.info("Downloading stock data for the first time. This may take a few minutes.")

    # force silencing of verbose API
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            for i in range(offset, end):
                s = symbols[i]
                data = yf.download(s, period=period)
                if len(data.index) == 0:
                    continue

                is_valid[i] = True
                data.to_csv(join(stocks_dir, '{}.csv'.format(s)))

    print('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))


def data_cleaning(data):
    st.info("Cleaning Data. This should only take a few seconds.")

    # Make a duplicate of the data
    cleaned_data = data
    
    # Remove rows with missing values
    cleaned_data = cleaned_data.dropna()
    
    # Remove rows with negative values for stock prices and volume
    cleaned_data = data[(cleaned_data[['Open', 'High', 'Low', 'Close', 'Adj Close']] > 0).all(axis=1)]
    cleaned_data = data[cleaned_data['Volume'] > 0]
    
    # Remove duplicate rows
    cleaned_data = cleaned_data.drop_duplicates()
   
    # Remove rows where the 'Open' price is greater than the 'High' price
    cleaned_data = cleaned_data[cleaned_data['Open'] <= cleaned_data['High']]

    # Remove rows where the 'Open' price is less than the 'Low' price
    cleaned_data = cleaned_data[cleaned_data['Open'] >= cleaned_data['Low']]

    # Remove rows where the 'Close' price is greater than the 'High' price
    cleaned_data = cleaned_data[cleaned_data['Close'] <= cleaned_data['High']]

    # Remove rows where the 'Close' price is less than the 'Low' price
    cleaned_data = cleaned_data[cleaned_data['Close'] >= cleaned_data['Low']]

    # Remove rows where 'Low' > 'High'
    cleaned_data = cleaned_data[cleaned_data['Low'] <= cleaned_data['High']]

    # Remove rows where 'High' < 'Low'
    cleaned_data = cleaned_data[cleaned_data['High'] >= cleaned_data['Low']]

    # Calculate IQR for each numerical column
    iqr_values = cleaned_data.quantile(0.75) - cleaned_data.quantile(0.25)

    # Define the lower and upper bounds for outliers
    lower_bounds = cleaned_data.quantile(0.25) - 3 * iqr_values
    upper_bounds = cleaned_data.quantile(0.75) + 3 * iqr_values

    outlier_mask = ~((cleaned_data < lower_bounds) | (cleaned_data > upper_bounds)).any(axis=1)
    cleaned_data = cleaned_data[outlier_mask]

    return cleaned_data


def load_stock_data(selected_stock):
    st.info("Loading Data. This should only take a few seconds.")

    if not os.path.exists(stocks_dir):
        # Download stock data if the stocks directory does not exist
        download_stock_data()

    for filename in os.listdir(stocks_dir):
        file_path = join(stocks_dir, filename)

        if isfile(file_path) and filename.endswith(".csv"):
            stock_data = pd.read_csv(file_path)

            # Add Ticker column based on the filename (excluding '.csv')
            stock_data['Ticker'] = filename[:-4]

            # Convert 'Date' column to datetime format
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])

            # Append the data to the list
            dataframes.append(stock_data)

    data = pd.concat(dataframes, ignore_index=True)
    data.set_index(['Ticker', 'Date'], inplace=True)
    data.sort_index(inplace=True)

    try:
        # Use the downloaded S&P 500 data
        selected_stock_data = data.loc[selected_stock]
        # Call data_cleaning function on the selected stock data
        cleaned_data = data_cleaning(selected_stock_data)
        return cleaned_data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    

def color_suggestion(suggestion):
    if suggestion == "Buy":
        return "green"
    elif suggestion == "Sell":
        return "red"
    else:
        return "gray"


# Function to get real-time stock data
def realtime_stock_data(api_key, symbol):
    try:
        realtime_stock_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}'
        response = requests.get(realtime_stock_url)
        stock_data = response.json()
        return stock_data.get('Time Series (5min)', {})
    except Exception as e:
        return None


# Define a function to fetch top news stories
def get_top_news(selected_stock):
    news_url = f'https://newsapi.org/v2/everything?q={selected_stock}&language=en&apiKey={news_api_key}'

    try:
        response = requests.get(news_url)
        news_data = response.json()

        if news_data.get('status') == 'ok' and news_data.get('totalResults') > 0:
            articles = news_data['articles']

            # Display the top news stories
            for article in articles[:5]:  # Display the top 5 news stories
                st.markdown(f"**Title:** {article['title']}")
                st.write(f"**Description:** {article['description']}")
                st.markdown(f"**Source:** {article['source']['name']}")
                st.write(f"**Published At:** {article['publishedAt']}")
                st.write(f"**URL:** {article['url']}")
                st.write("---")
        else:
            st.warning("The market is currently closed. Try again later")

    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")


def plot_raw_data():

    fig = go.Figure()

    # Use the index for the x-axis
    last_365_days_data = data.iloc[-365:]

    # Plot stock open and close prices
    fig.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['Close'], name="stock_close"))

    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # Plot RSI
    st.subheader('RSI (Relative Strength Index)')
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['RSI'], name="RSI"))
    fig_rsi.layout.update(title_text='RSI Indicator', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_rsi)

    # Plot Moving Averages
    st.subheader('Moving Averages')
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['SMA50'], name="50-day SMA"))
    fig_ma.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['SMA200'], name="200-day SMA"))
    fig_ma.layout.update(title_text='Moving Averages', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_ma)

    # Plot MACD
    st.subheader('MACD (Moving Average Convergence Divergence)')
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['MACD'], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['Signal_Line'], name="Signal Line"))
    fig_macd.add_trace(go.Scatter(x=last_365_days_data.index.get_level_values('Date'), y=last_365_days_data['MACD_Histogram'], name="MACD Histogram"))
    fig_macd.layout.update(title_text='MACD Indicator', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_macd)


def predict_stock_prices(data):
    # Select the last 365 days of data
    last_year_data = data.reset_index().tail(365)[['Date', 'Close']]
    last_year_data.columns = ['ds', 'y']

    # Initialize the Prophet model
    model = Prophet()
    model.fit(last_year_data)
    future = model.make_future_dataframe(periods=30)  # Predicting for the next 30 days
    forecast = model.predict(future)

    return model, forecast

##### END FUNCTIONS SECTION #####


##### MAIN APP #####

def main():
    global data  # Declare data as a global variable
    st.title('MarketSense Pro')

    # Use a text input for the stock ticker symbol
    selected_stock = st.text_input('Enter stock ticker symbol for prediction', 'AAPL')

    # button to trigger data loading and analysis
    if st.button("Load Data and Analyze"):
        # Load data
        # @st.cache_data

        data = load_stock_data(selected_stock)

        if data is not None:
            st.subheader('Raw data')
            st.write(data.tail())

            # Calculate RSI
            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

            # Calculate moving averages
            data['SMA50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
            data['SMA200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()

            # Calculate MACD using the provided pandas approach
            k = data['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
            d = data['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
            macd = k - d
            macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
            macd_h = macd - macd_s

            # Add MACD data to the DataFrame
            data['MACD'] = macd
            data['Signal_Line'] = macd_s
            data['MACD_Histogram'] = macd_h

            # Suggestion of whether to buy or sell based on the indicators
            data['Buy_Sell_Signal'] = "Hold"  # Default to "Hold"
            data.loc[(data['RSI'] < rsi_buy_threshold) & (data['MACD'] > data['Signal_Line']), 'Buy_Sell_Signal'] = "Buy"
            data.loc[(data['RSI'] > rsi_sell_threshold) & (data['MACD'] < data['Signal_Line']), 'Buy_Sell_Signal'] = "Sell"

            suggestion = data['Buy_Sell_Signal'].iloc[-1]
            box_color = color_suggestion(suggestion)

            # Display the suggestion
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown(f'<div style="background-color: {box_color}; padding: 10px"><b>{suggestion}</b></div>', unsafe_allow_html=True)

        # Display real-time stock data or a warning if the market is closed
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.subheader('Real-Time Stock Data')

        real_time_stock_data = realtime_stock_data(realtime_api_key, selected_stock)

        if real_time_stock_data:
            latest_data = real_time_stock_data[list(real_time_stock_data.keys())[0]]
            st.write(f"Real-Time Price ({selected_stock}): ${latest_data['4. close']}")
        else:
            st.warning("The market is closed and real-time stock data is not available. Try again when the market reopens")
            
        # Plot raw data, RSI, moving averages, and MACD
        plot_raw_data()

        # Predict next year's stock prices using Prophet
        result = predict_stock_prices(data)
        model = result[0]
        forecast = result[1]    

        # Display the forecast data and plot
        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for 1 year')
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1)

        # Create section for news
        st.subheader('Top News')

        # Call the function to get top news
        get_top_news(selected_stock)

if __name__ == "__main__":
    main()