#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import os
from absl import app, flags
from tqdm import tqdm


flags.DEFINE_string('period', 'max', 'Time period for price data')
flags.DEFINE_string('interval', '1d', 'Price data resolution')
flags.DEFINE_string('ticker_file', 'data/tickers.csv',
                    'File to store tickers in')
flags.DEFINE_string('closes_dir', 'data',
                    'Directory for closes of all tickers')
flags.DEFINE_string('price_dir', 'data/prices',
                    'Directory for storing full ticker data')
FLAGS = flags.FLAGS


def main(_) -> None:
    # Load tickers
    tickers = pd.read_csv(FLAGS.ticker_file)
    # Load prices and dump to disk
    os.makedirs(f'{FLAGS.price_dir}/{FLAGS.interval}', exist_ok=True)
    closes = {}
    for ticker in tqdm(tickers.ticker):
        data = yf.Ticker(ticker)
        history = data.history(period=FLAGS.period,
                               interval=FLAGS.interval)
        # Full history to a separate csv for each ticker
        history.to_csv(f'{FLAGS.price_dir}/{FLAGS.interval}/{ticker}.csv')
        # Collect closes for all tickers
        closes[ticker] = history.Close
    # Convert to dataframe
    closes = pd.DataFrame(closes).sort_index()
    closes.to_csv(f'{FLAGS.closes_dir}/closes_{FLAGS.interval}.csv')


if __name__ == '__main__':
    app.run(main)
