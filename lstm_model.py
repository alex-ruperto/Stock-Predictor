import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.impute import SimpleImputer  # Add this import

# features
def preprocess_data(df):
    time_interval = 26 # adjust this to whatever the time interval from raw_data in data_processing is. there are 26 15 minute interval candles in a single trading day.
    # compare next day closing price to current day. convert boolean values to integer values
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # df['Close] is closing price for each candle. target is to get next close higher than current.
    df['SMA1'] = df['Close'].rolling(window=50*time_interval).mean()
    df['SMA2'] = df['Close'].rolling(window=200*time_interval).mean()
    df['RSI'] = rsi_formula(df['Close'], window=14*time_interval)
    df['MACD_Line'], df['Signal_Line'] = macd_formula(df['Close'], short_window=12*time_interval, long_window=26*time_interval, signal_window=9*time_interval)
    df['Upper_Bollinger'], df['Lower_Bollinger'] = bollinger_bands(df['Close'], window=20*time_interval)
    df['K_Line'], df['D_Line'] = stochastic_oscillator(df['Close'], window=14*time_interval)

    return df

def rsi_formula(data, window):
    """Compute the RSI for a given data series."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd_formula(data, short_window, long_window, signal_window):
    """Compute the MACD for a given data series."""
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    return macd_line, signal_line

def bollinger_bands(data, window, num_std_dev=2):
    """Compute Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def stochastic_oscillator(data, window):
    """Compute Stochastic Oscillator"""
    low_min = data.rolling(window=window).min()
    high_max = data.rolling(window=window).max()
    k_line = 100 * ((data - low_min) / (high_max - low_min))
    d_line = k_line.rolling(window=3).mean()
    return k_line, d_line

# end of features

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
        y_pred = self.sigmoid(self.linear(lstm_out[:, -1, :]))
        return y_pred


# model training and data pre processing
def train_model(df):
    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df_imputed = df.copy()
    df_imputed[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 
                'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 
                'D_Line']] = imputer.fit_transform(df[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 
                'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line']])
    
    # Split data into training and test sets
    X = df_imputed[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line']]
    y = df_imputed['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Reshape data for LSTM
    # In this case, using a lookback of 60 (similar to your rolling window in backtrader)
    # Reshape to (samples, time_steps, features)
    X_train_reshaped = np.array([X_train.values[i-60:i] for i in range(60, X_train.shape[0])])
    y_train_reshaped = y_train.values[60:]
    X_test_reshaped = np.array([X_test.values[i-60:i] for i in range(60, X_test.shape[0])])
    y_test_reshaped = y_test.values[60:]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_reshaped[:, None], dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_reshaped[:, None], dtype=torch.float32)

    model = LSTMModel(X_train_reshaped.shape[2], 50)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(15): # update model's weights 15 times. each epoch, it will start with the weights of the previous epoch. the hope is to reduce the amount of loss each time.
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor) # how well the model is perfoming during the dataset. consistently decreasing training may be good but also may mean it's overfitting.
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor) # how well does the model perform on the dataset that it hasn't seen during training. validation set.
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    y_pred = model(X_test_tensor)
    y_pred = (y_pred.detach().numpy() > 0.5).astype(int)
    accuracy = accuracy_score(y_test_reshaped, y_pred)
    print(f"Model Accuracy on Test Set with LSTM: {accuracy:.2f}")
    
    return model
    
    

