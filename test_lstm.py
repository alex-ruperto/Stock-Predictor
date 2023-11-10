import unittest
from lstm_model import preprocess_data, train_model, prepare_data_for_lstm
import yfinance as yf 
from datetime import datetime, timedelta
import torch
from sklearn.model_selection import train_test_split
import math

class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_interval = 6.5 # this is the actual time interval used in the data.
        ticker = "AMC"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
        cls.data = yf.download(ticker, start = start_date, end = end_date, interval='1h', auto_adjust=True)
    
    # def test_preprocess_data(self):
    #     warm_up_period = math.ceil(200*self.time_interval)
    #     # Apply the preprocessing
    #     processed_data = preprocess_data(self.data)

    #     # Check that the returned DataFrame is not empty
    #     self.assertTrue(processed_data.iloc[:warm_up_period].isnull().values.any(), "Expected NaN values in warm-up period")
    #     self.assertFalse(processed_data.iloc[warm_up_period:].isnull().values.any(), "Unexpected NaN values found after the warm-up period")


    
    # def test_data_quality(self):
    #     # Convert the index to a column to check for missing date-times
    #     self.data.reset_index(inplace=True)

    #     # Check for missing values
    #     missing_values = self.data.isnull().sum()
    #     self.assertFalse(missing_values.any(), "There are missing values in the dataset")

    #     # Check for duplicates
    #     duplicate_rows = self.data[self.data.duplicated()]
    #     self.assertTrue(duplicate_rows.empty, "There are duplicate rows in the dataset")


    #     # Statistical summary to check for outliers and inconsistencies
    #     statistical_summary = self.data.describe()
    #     print(f"Statistical summary:\n{statistical_summary}")

    #     # Check for the correct data type of each column except for the datetime one.
    #     numeric_columns = self.data.select_dtypes(include=['float64', 'int64'])
    #     self.assertTrue(numeric_columns.shape[1] == self.data.shape[1] - 1, "Not all data types are numeric")


    def test_test_accuracy(self):
            df = preprocess_data(self.data)
            X, y = df[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line', 'EMA1', 'EMA2', 'OBV_Scaled', 'VWAP', 'Volatility', 'ROC', 'ATR']], df['Target']
            X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_data_for_lstm(X, y)

            model, best_threshold = train_model(df, epochs=30)

            model.eval()
            with torch.no_grad():
                test_predictions_logits = model(X_test_tensor).squeeze()
                test_predictions = torch.sigmoid(test_predictions_logits)
                best_test_predictions = (test_predictions > best_threshold).int()
                correct_test_predictions = (best_test_predictions == y_test_tensor.int()).float().sum()
                test_accuracy = correct_test_predictions / len(y_test_tensor)

            # Convert tensors to numpy arrays for easy handling
            test_predictions_np = best_test_predictions.numpy()
            y_test_np = y_test_tensor.squeeze().numpy()

            # Print the first 50 predictions and actual values
            print("First 100 Predictions:", test_predictions_np[:100])
            print("First 100 Actual Values:", y_test_np[:100])

            # Assert that the test accuracy is above a certain threshold
            self.assertGreater(test_accuracy.item(), 0.5, "Test accuracy should be greater than 50%")

    

if __name__ == '__main__':
    unittest.main()