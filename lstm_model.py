import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import pandas_ta

time_interval = 6.5 # Adjust this to whatever the time interval from raw_data in data_processing is. There are 6.5 hourly interval candles in a single trading day.

# features
def preprocess_data(df):
    # compare next day closing price to current day. convert boolean values to integer values
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # df['Close] is closing price for each candle. target is to get next close higher than current.
    # Simple Moving Averages 1 and 2
    df['SMA1'] = pandas_ta.sma(df['Close'], length=50*time_interval)
    df['SMA2'] = pandas_ta.sma(df['Close'], length=200*time_interval)
    
    # Relative Strength Index (RSI)
    df['RSI'] = pandas_ta.rsi(df['Close'], length=14*time_interval)

    # Moving Average Convergence Divergence (MACD)
    macd = pandas_ta.macd(df['Close'], fast=12*time_interval, slow=26*time_interval, signal=9*time_interval)
    macd_line_col = macd.columns[0]  # MACD line is typically the first column
    signal_line_col = macd.columns[2]  # Signal line is typically the third column
    df['MACD_Line'] = macd[macd_line_col]
    df['Signal_Line'] = macd[signal_line_col]

    # Bollinger Bands
    bollinger = pandas_ta.bbands(df['Close'], length=20*time_interval, std=2)
    upper_band_col = bollinger.columns[0]  # Upper band is typically the first column
    lower_band_col = bollinger.columns[2]  # Lower band is typically the third column
    df['Upper_Bollinger'] = bollinger[upper_band_col]
    df['Lower_Bollinger'] = bollinger[lower_band_col]

    # Stochastic Oscillator
    # Ensure k and d are integers
    k = int(14 * time_interval)
    d = int(3 * time_interval)
    stoch = pandas_ta.stoch(df['High'], df['Low'], df['Close'], k=k, d=d)
    # 14 is the look back period, 3 is period for %K smoothing, 3 for %D line. %D is moving average of %K
    # multiply those numbers by the time interval
    k_line_col = stoch.columns[0]  # %K line is typically the first column
    d_line_col = stoch.columns[1]  # %D line is typically the second column
    df['K_Line'] = stoch[k_line_col]
    df['D_Line'] = stoch[d_line_col]

    # Exponential Moving Averages
    df['EMA1'] = pandas_ta.ema(df['Close'], length=12*time_interval)
    df['EMA2'] = pandas_ta.ema(df['Close'], length=26*time_interval)

    # On-Balance Volume
    df['OBV'] = pandas_ta.obv(df['Close'], df['Volume'])
    scaler = StandardScaler()
    df['OBV_Scaled'] = scaler.fit_transform(df[['OBV']])
    df = df.drop('OBV', axis=1)

    # Volume-weighted Average Price (VWAP)
    df['VWAP'] = pandas_ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], length=14*time_interval)

    # Historical Volatility
    df['Volatility'] = pandas_ta.stdev(df['Close'], length=14*time_interval)

    # Price Rate of Change
    df['ROC'] = pandas_ta.roc(df['Close'], length=10*time_interval)

    # Average True Range
    df['ATR'] = pandas_ta.atr(df['High'], df['Low'], df['Close'], length=14*time_interval)

    return df

# end of features

def prepare_data_for_lstm(X, y, test_size=0.25, random_state=42, lookback=60):
    ######################### Split Data into Training and Test Sets then Reshape for LSTM #########################
    # Split data into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_state)  

    # Reshape data for LSTM
    X_train_reshaped = np.array([X_train[i-lookback:i] for i in range(lookback, X_train.shape[0])])
    y_train_reshaped = y_train[lookback:]
    X_val_reshaped = np.array([X_val[i-lookback:i] for i in range(lookback, X_val.shape[0])])
    y_val_reshaped = y_val[lookback:]
    X_test_reshaped = np.array([X_test[i-lookback:i] for i in range(lookback, X_test.shape[0])])
    y_test_reshaped = y_test[lookback:]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_reshaped.to_numpy()[:, None], dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_reshaped, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_reshaped.to_numpy()[:, None], dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_reshaped.to_numpy()[:, None], dtype=torch.float32)
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

# note: a tensor is a data structure that is a multi-dimensional array that can hold scalars, vectors, matrices, or higher-dimensional data.
# model. Read sequence of feature vectors and process them with the LSTM layer. Produce a singlee 0 and 1 for each input sequence using the fully connected layer and sigmoid activation function.
class LSTMModel(nn.Module): # nn module is the base class for all neural networks modules in PyTorch.
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3): # constructor. input_dim = size of input feature vector, hidden_dim = number of hidden units in LSTM layer.
        super(LSTMModel, self).__init__() # call the init function from the superclass.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True) # expects input tensors of shape (batch, seq_len, input_dim), and it outputs (batch, seq_len, hidden_dim).
        
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() # squashes the output between 0 and 1. value will always be between 0 and 1.

    def forward(self, x): # defines forward pass of neural network.
        lstm_out, _ = self.lstm(x)
        # handle a batch size of 1
        if lstm_out.dim() == 2:
            lstm_out = lstm_out.unsqueeze(0)
        y_pred = self.sigmoid(self.linear(lstm_out[:, -1, :]))
        return y_pred

'''
------------------------------------------------Train Validate Test------------------------------------------------
'''
def train_model(df, epochs=30, batch_size=64, learning_rate=0.1):
    # Define warm-up periods for each feature
    feature_warm_up_periods = {
        'Target': 0,
        'SMA1': int(50 * time_interval),
        'SMA2': int(200 * time_interval),
        'RSI': int(14 * time_interval),
        'MACD_Line': int(50 * time_interval),
        'Signal_Line': int(9 * time_interval),
        'Upper_Bollinger': int(20 * time_interval),
        'Lower_Bollinger': int(20 * time_interval),
        'K_Line': int(14 * time_interval),
        'D_Line': int(3 * time_interval),
        'EMA1': int(12 * time_interval),
        'EMA2': int(26 * time_interval),
        'OBV_Scaled': 0,
        'VWAP': int(14 * time_interval),
        'Volatility': int(14 * time_interval),
        'ROC': int(10 * time_interval),
        'ATR': int(14 * time_interval)
    }

    def dynamically_introduce_features(df, feature_periods):
        max_warm_up = max(feature_periods.values())
        dynamic_df = pd.DataFrame(index=df.index[max_warm_up:])

        for feature, warm_up in feature_periods.items():
            dynamic_df[feature] = df[feature].iloc[warm_up:]
        
        return dynamic_df.dropna()

    def find_best_threshold(y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold  # Return only the threshold value
    
    df_dynamic = dynamically_introduce_features(df, feature_warm_up_periods)
    
    X = df_dynamic[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line', 'EMA1', 'EMA2', 'OBV_Scaled', 'VWAP', 'Volatility', 'ROC', 'D_Line']]
    y = df_dynamic['Target']

    # Prepare the data
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_data_for_lstm(X, y)
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


######################### Create, Train, Validate and Test LSTM Model #########################
    model = LSTMModel(X_train_tensor.shape[-1], 200)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    for epoch in range(epochs): # update model's weights (epochs) times. each epoch, it will start with the weights of the previous epoch. the hope is to reduce the amount of loss each time.
        model.train() # set the model to train mode.
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate for hyperparameter tuning
        model.eval()
        with torch.no_grad():
            val_predictions_logits = model(X_val_tensor).squeeze()
            val_predictions = torch.sigmoid(val_predictions_logits)
            y_val_binary_np = y_val_tensor.int().numpy()

            # Find the best threshold using ROC curve analysis
            best_threshold = find_best_threshold(y_val_binary_np, val_predictions.numpy())

            # Calculate accuracy using the best threshold
            y_val_binary_squeezed = y_val_tensor.squeeze()
            best_binary_predictions = (val_predictions > best_threshold).int()
            correct_val_predictions = (best_binary_predictions == y_val_binary_squeezed).float().sum()
            val_accuracy = correct_val_predictions / y_val_binary_squeezed.size(0)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Accuracy: {val_accuracy:.4f}, Best Threshold: {best_threshold:.2f}, Correct val predictions: {correct_val_predictions}, y-val binary size: {y_val_binary_squeezed.size(0)}")

    # Calculate test accuracy
    model.eval() 
    with torch.no_grad():
        test_predictions_logits = model(X_test_tensor).squeeze()
        test_predictions = torch.sigmoid(test_predictions_logits)

        # Use best threshold determined during the training
        best_test_predictions = (test_predictions > best_threshold).int()

        # Ensure the tensors are of the same shape
        y_test_tensor_squeezed = y_test_tensor.squeeze()
        
        # Calculate correct predictions
        correct_test_predictions = (best_test_predictions == y_test_tensor_squeezed).float()

        # Sum and calculate accuracy
        num_correct_test_predictions = correct_test_predictions.sum()
        print(f"Number of Correct Test Predictions: {num_correct_test_predictions}, Length of y_test_tensor: {len(y_test_tensor_squeezed)}")
        test_accuracy = num_correct_test_predictions / len(y_test_tensor_squeezed)

        print(f"Test Accuracy: {test_accuracy.item():.4f}")
    
    return model, best_threshold