import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.impute import SimpleImputer  # Add this import
from pandas import concat
import pandas_ta

# features
def preprocess_data(df):
    time_interval = 6.5 # adjust this to whatever the time interval from raw_data in data_processing is. there are 26 15 minute interval candles in a single trading day.
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
    
    return df

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
    
    

