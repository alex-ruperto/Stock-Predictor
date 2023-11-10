import unittest
from lstm_model import preprocess_data
import pandas as pd
import numpy as np
import yfinance as yf 
from datetime import datetime, timedelta
import pandas_ta
from lstm_model import train_model
import torch
from sklearn.model_selection import train_test_split

class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_interval = 6.5 # this is the actual time interval used in my data.
        ticker = "AAPL"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
        cls.stock_data = yf.download(ticker, start = start_date, end = end_date, interval='1h', auto_adjust=True)
        cls.df = preprocess_data(cls.stock_data)

    def setUp(self):
        # Split your data into training, validation, and test sets here
        # This code assumes that `self.stock_data` has already been preprocessed
        # and is ready to be split into features and labels
        X = self.stock_data.drop('Target', axis=1).values
        y = self.stock_data['Target'].values
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.25, random_state=42)

        # Convert the validation set to tensors
        self.X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32)

    def test_model_accuracy(self):
        # Train the model and get the best threshold
        model, best_threshold, reported_accuracy = train_model(self.df, epochs=50)

        # Set model to evaluation mode and get predictions
        model.eval()
        with torch.no_grad():
            val_predictions = model(self.X_val_tensor).squeeze()

        # Apply the threshold to get binary predictions
        val_predictions_binary = (val_predictions > best_threshold).int()

        # Calculate accuracy manually
        correct_predictions = (val_predictions_binary == self.y_val_tensor.int()).float().sum()
        manual_accuracy = correct_predictions / len(self.y_val_tensor)

        # Compare reported accuracy to manual accuracy
        self.assertAlmostEqual(manual_accuracy.item(), reported_accuracy, places=4,
                               msg=f"Reported accuracy ({reported_accuracy}) does not match manually calculated accuracy ({manual_accuracy.item()})")

if __name__ == '__main__':
    unittest.main()