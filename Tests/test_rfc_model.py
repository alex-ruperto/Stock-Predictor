import unittest
from unittest.mock import patch
from random_forest_model import preprocess_data # import this function for testing
from random_forest_model import train_random_forest_model
import yfinance as yf 
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
# TODO write a unit test for random_forest_model.py

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_interval = 6.5 # this is the actual time interval used in the data.
        ticker = "GOOG"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
        cls.data = yf.download(ticker, start = start_date, end = end_date, interval='1h', auto_adjust=True)
        cls.data.columns = cls.data.columns.str.lower() # make sure all columns are lowercase

        
    
    @patch('random_forest_model.logger') # patch the logger used in random_forest_model
    def test_logging_and_nans(self, mock_logger):
        preprocessed_data = preprocess_data(self.data)

        self.assertTrue(mock_logger.info.called)
        self.assertFalse(mock_logger.warning.called)

        mock_logger.info.assert_any_call("Data preprocessing started.")
        mock_logger.info.assert_any_call("Data preprocessing complete.")

    @patch('random_forest_model.logger') 
    def test_model_training(self, mock_logger):
        preprocessed_data = preprocess_data(self.data) # preprocess the data before calling the train function
        model = train_random_forest_model(preprocessed_data)
        
        self.assertTrue(mock_logger.info.called)
        
if __name__ == '__main__':
    unittest.main()