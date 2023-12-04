import pandas as pd
import yfinance as yf
import os, contextlib
import shutil
from os.path import isfile, join
from datetime import date


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


offset = 0
limit = 3000
period = 'max' # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

stocks_dir = "stocks"
dataframes = []


# symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
# limit = limit if limit else len(symbols)
# end = min(offset + limit, len(symbols))
# is_valid = [False] * len(symbols)
# # force silencing of verbose API
# with open(os.devnull, 'w') as devnull:
#     with contextlib.redirect_stdout(devnull):
#         for i in range(offset, end):
#             s = symbols[i]
#             data = yf.download(s, period=period)
#             if len(data.index) == 0:
#                 continue
        
#             is_valid[i] = True
#             data.to_csv('hist/{}.csv'.format(s))

# print('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))


# Loop through each file in the 'stocks' directory
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


combined_data = pd.concat(dataframes, ignore_index=True)

combined_data.set_index(['Ticker', 'Date'], inplace=True)

combined_data.sort_index(inplace=True)


print(combined_data.head())