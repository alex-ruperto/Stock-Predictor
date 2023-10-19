import unittest
from lstm_model import preprocess_data, macd_formula, rsi_formula, stochastic_oscillator, bollinger_bands
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class TestFeatures(unittest.TestCase):
    time_interval = 26 # this is the actual time interval used in my data.
    ticker = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start = start_date, end = end_date, interval='15m', auto_adjust=True)

    def test_macd_formula(self):
        print("Running MACD Formula unit test...")
        # Use real stock data
        data = self.stock_data['Close']
        
        # Call function
        macd_line, signal_line = macd_formula(data, short_window=12, long_window=26, signal_window=9)
        expected_macd_line, expected_signal_line = macd_formula(data, short_window=12*self.time_interval, long_window=26*self.time_interval, signal_window=9*self.time_interval)

        # Check results
        pd.testing.assert_series_equal(macd_line, expected_macd_line, check_exact=False)
        pd.testing.assert_series_equal(signal_line, expected_signal_line, check_exact=False)



    def test_rsi_formula(self):
        print("\nRunning RSI Formula unit test...")
        data = pd.Series(np.random.randn(100), name = 'Close') # generate random data

        RSI = rsi_formula(data, window=14)
        RSI2 = rsi_formula(data, window=14*self.time_interval)

        # Check results
        self.assertEqual(len(RSI), len(data))
        self.assertEqual(len(RSI2), len(data))

    def test_stochastic_oscillator(self):
        print("\nRunning Stochastic Oscillator Formula unit test...")
        data = pd.Series(np.random.randn(100), name = 'Close') # generate random data

        RSI = rsi_formula(data, window=14)
        RSI2 = rsi_formula(data, window=14*self.time_interval)

        # Check results
        self.assertEqual(len(RSI), len(data))
        self.assertEqual(len(RSI2), len(data))



    def test_preprocess_data(self):
        print("\nRunning Preprocess Data function unit test...")
        # Your test code here...

if __name__ == '__main__':
    unittest.main()