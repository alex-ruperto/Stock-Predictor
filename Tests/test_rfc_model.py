import unittest
import yfinance as yf 
import pandas as pd
import numpy as np
from Models.random_forest_model import RandomForestTrainer
from Preprocessors.rfc_preprocessor import RFCPreprocessor
from Utils.data_processing import backtest
from unittest.mock import patch
from datetime import datetime, timedelta
# TODO write a unit test for random_forest_model.py

class TestRFCModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_interval = 6.5 # this is the actual time interval used in the data.
        cls.ticker = "GOOG"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
        cls.data = yf.download(cls.ticker, start = start_date, end = end_date, interval='1h', auto_adjust=True)
        cls.data.columns = cls.data.columns.str.lower() # make sure all columns are lowercase

    def test_preprocessing(self):
        # Test data preprocessing
        preprocessor = RFCPreprocessor()
        preprocessed_data = preprocessor.preprocess(self.data)
        self.assertIn("sma1", preprocessed_data.columns)
        self.assertIn("target", preprocessed_data.columns)
        self.assertFalse(preprocessed_data.isnull().values.any())
    
    @patch('Utils.logger_config.configure_logger')  # Patch the logger configuration function
    def test_model_training(self, mock_logger):  # Add the mock_logger argument
        # Test Random Forest training
        preprocessor = RFCPreprocessor()
        preprocessed_data = preprocessor.preprocess(self.data)
        trainer = RandomForestTrainer()
        model = trainer.train(preprocessed_data, "target")
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
    
    def test_backtest(self):
        # Test backtesting functionality
        results = backtest(self.ticker)
        self.assertIsInstance(results, tuple)
        self.assertTrue(len(results) > 0)
        
if __name__ == '__main__':
    unittest.main()