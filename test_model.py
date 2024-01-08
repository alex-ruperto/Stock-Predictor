import unittest
from random_forest_model import preprocess_data
import yfinance as yf 
from datetime import datetime, timedelta
import pandas as pd

# TODO write a unit test for random_forest_model.py
class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_interval = 6.5 # this is the actual time interval used in the data.
        ticker = "GOOG"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
        cls.data = yf.download(ticker, start = start_date, end = end_date, interval='1h', auto_adjust=True)

    def test_preprocess_data(self):
        # Call the preprocess_data function
        processed_data = preprocess_data(self.data)

        # Check if the returned object is a DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame)

        # Check for the presence of expected columns
        expected_columns = ['Target', 'SMA1', 'SMA2', 'RSI', 'EMA1', 'EMA2', 'Volatility', 'ROC', 'ATR']
        for column in expected_columns:
            self.assertIn(column, processed_data.columns)

        # Check for no missing values in the DataFrame
        self.assertFalse(processed_data.isnull().values.any())

        # You can add more tests here to check for data types, value ranges, etc.

if __name__ == '__main__':
    unittest.main()