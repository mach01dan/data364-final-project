import streamlit as st
from datetime import date, time, datetime
import yfinance as yf
from prophet import Prophet
import requests
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import ta

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Use a text input for the stock ticker
selected_stock = st.text_input('Enter stock ticker symbol for prediction', 'AAPL')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Set threshold values for the indicators
rsi_buy_threshold = 45  # Lower RSI values suggest buying opportunities
rsi_sell_threshold = 55  # Higher RSI values suggest selling opportunities
macd_buy_threshold = 0  # Higher MACD values relative to the signal line suggest buying opportunities
macd_sell_threshold = 0  # Lower MACD values relative to the signal line suggest selling opportunities

# Use a button to trigger data loading and analysis
if st.button("Load Data and Analyze"):
    # Load data
    @st.cache_data
    def load_data(selected_stock=selected_stock):
        try:
            data = yf.download(selected_stock, START, TODAY)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    data = load_data(selected_stock)

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

        # Create a suggestion of whether to buy or sell based on the indicators
        data['Buy_Sell_Signal'] = "Hold"  # Default to "Hold"
        data.loc[(data['RSI'] < rsi_buy_threshold) & (data['MACD'] > data['Signal_Line']), 'Buy_Sell_Signal'] = "Buy"
        data.loc[(data['RSI'] > rsi_sell_threshold) & (data['MACD'] < data['Signal_Line']), 'Buy_Sell_Signal'] = "Sell"

        # Determine the color of the suggestion box
        def color_suggestion(suggestion):
            if suggestion == "Buy":
                return "green"
            elif suggestion == "Sell":
                return "red"
            else:
                return "gray"

        suggestion = data['Buy_Sell_Signal'].iloc[-1]
        box_color = color_suggestion(suggestion)

        # Display the suggestion in a colored box
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(f'<div style="background-color: {box_color}; padding: 10px"><b>{suggestion}</b></div>', unsafe_allow_html=True)

        
        
        stock_api_key = 'IDKN0O488FIDEPVR'

    # Function to check if the market is currently open
    def is_market_open():
        now = datetime.now().time()
        market_open_time = time(9, 30)
        market_close_time = time(16, 0)
        return market_open_time <= now <= market_close_time

    # Function to get real-time stock data
    def get_real_time_stock_data(api_key, symbol):
        if is_market_open():
            try:
                stock_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}'
                response = requests.get(stock_url)
                stock_data = response.json()
                return stock_data.get('Time Series (5min)', {})
            except Exception as e:
                return None
        else:
            return None

    # Display real-time stock data or a warning if the market is closed
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.subheader('Real-Time Stock Data')


    real_time_stock_data = get_real_time_stock_data(stock_api_key, selected_stock)

    if real_time_stock_data:
        latest_data = real_time_stock_data[list(real_time_stock_data.keys())[0]]
        st.write(f"Real-Time Price ({selected_stock}): ${latest_data['4. close']}")
    else:
        st.warning("The market is closed and real-time stock data is not available. Try again when the market reopens")
        
        # Plot raw data, RSI, moving averages, and MACD
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

            st.subheader('RSI (Relative Strength Index)')
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name="RSI"))
            fig_rsi.layout.update(title_text='RSI Indicator', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_rsi)

            st.subheader('Moving Averages')
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA50'], name="50-day SMA"))
            fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA200'], name="200-day SMA"))
            fig_ma.layout.update(title_text='Moving Averages', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_ma)

            st.subheader('MACD (Moving Average Convergence Divergence)')
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name="MACD"))
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal_Line'], name="Signal Line"))
            fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD_Histogram'], name="MACD Histogram"))
            fig_macd.layout.update(title_text='MACD Indicator', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_macd)

        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)


    # Add a new section for news
    st.subheader('Top News')

    # Define a function to fetch top news stories
    def get_top_news(selected_stock):
        api_key = '60d16a8cc922453896de9495f9b99b1d'  # Get your News API key
        news_url = f'https://newsapi.org/v2/everything?q={selected_stock}&apiKey={api_key}'

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
                st.warning("No news available for this stock.")

        except Exception as e:
            st.error(f"An error occurred while fetching news: {e}")

    # Call the function to get top news
    get_top_news(selected_stock)