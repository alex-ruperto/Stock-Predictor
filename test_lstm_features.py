import unittest
from lstm_model import preprocess_data
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta

class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_interval = 6.5 # this is the actual time interval used in my data.
        ticker = "AAPL"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
        cls.stock_data = yf.download(ticker, start = start_date, end = end_date, interval='1h', auto_adjust=True)

    def test_macd_formula(self):
        print("Running MACD Formula unit test...")
        # Use real stock data
        data = self.stock_data['Close'].copy()

        # Calculate MACD using pandas_ta
        macd_ta = pandas_ta.macd(data, fast=12*self.time_interval, slow=26*self.time_interval, signal=9*self.time_interval)

        # Dynamically retrieve the MACD and Signal line column names
        macd_line_col = macd_ta.columns[0]  # MACD line is typically the first column
        signal_line_col = macd_ta.columns[2]  # Signal line is typically the third column

        # pandas_ta method returns dataframe with macd, macd_signal, and macd_histogram
        # Use the dynamic column names for expected values
        expected_macd_line = macd_ta[macd_line_col]
        expected_signal_line = macd_ta[signal_line_col]
        
        # Get MACD and Signal Line from processed data
        processed_data = preprocess_data(self.stock_data.copy())
        macd_line = processed_data['MACD_Line']
        signal_line = processed_data['Signal_Line']

        # Check results
        pd.testing.assert_series_equal(macd_line, expected_macd_line, rtol=1e-5, atol=1e-8, check_names=False)
        pd.testing.assert_series_equal(signal_line, expected_signal_line, rtol=1e-5, atol=1e-8, check_names=False)

        # Note: rtol is the relative tolerance. 1e-5 is 0.001%. Ratio of difference between two values
        # compared to the larger of the two values being compared. 
        # atol is the maximum allowed tolerance. 1e-8 is 0.00000001%. 
        # it will ignore the names of the series.

if __name__ == '__main__':
    unittest.main()