#!/usr/bin/env python3
import requests
import os
import pandas as pd
from absl import app, flags
from bs4 import BeautifulSoup


SP500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


flags.DEFINE_string('ticker_file', 'data/tickers.csv',
                    'File to store tickers in')
FLAGS = flags.FLAGS


def sp500_tickers() -> list[str]:
    '''
    Downloads S&P-500 tickers from Wikipedia.
    '''
    page = requests.get(SP500_URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr'):
        ticker = row.findAll('td')
        # A row in table might not contain a ticker (column labels)
        if ticker:
            tickers.append(ticker[0].text.strip())
    tickers = sorted(tickers)
    return tickers


def commodity_tickers() -> list[str]:
    '''
    Downloads futures contract tickers for
    traded commodities from Wikipedia.
    '''
    tickers = []
    return tickers


def main(_) -> None:
    tickers = sp500_tickers()
    tickers += commodity_tickers()
    print(f'Found {len(tickers)} tickers.')
    print(f'Writing to {FLAGS.ticker_file}.')
    # Dump to file
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame({'ticker': tickers})
    df.to_csv(FLAGS.ticker_file, index=False)


if __name__ == '__main__':
    app.run(main)
