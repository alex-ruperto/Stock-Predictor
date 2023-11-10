import unittest
from lstm_model import preprocess_data, train_model, prepare_data_for_lstm
import yfinance as yf 
from datetime import datetime, timedelta
import torch
from sklearn.model_selection import train_test_split

class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_interval = 6.5 # this is the actual time interval used in the data.
        ticker = "GOOG"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')
        cls.data = yf.download(ticker, start = start_date, end = end_date, interval='1h', auto_adjust=True)
        
    def test_test_accuracy(self):
        df = preprocess_data(self.data)

        # Handle missing values using mean imputation and prepare data
        X, y = df[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line']], df['Target']
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_data_for_lstm(X, y)

        model, best_threshold = train_model(df, epochs=50)

        model.eval()
        with torch.no_grad():
            test_predictions_logits = model(X_test_tensor).squeeze()
            test_predictions = torch.sigmoid(test_predictions_logits)
            best_test_predictions = (test_predictions > best_threshold).int()
            correct_test_predictions = (best_test_predictions == y_test_tensor.int()).float().sum()
            test_accuracy = correct_test_predictions / len(y_test_tensor)

        # Assert that the test accuracy is above a certain threshold
        self.assertGreater(test_accuracy.item(), 0.5, "Test accuracy should be greater than 50%")

if __name__ == '__main__':
    unittest.main()