import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
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

def prepare_data_for_lstm(X, y, test_size=0.25, random_state=42, lookback=120):
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

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), 'checkpoint.pt')
            self.val_loss_min = val_loss


# note: a tensor is a data structure that is a multi-dimensional array that can hold scalars, vectors, matrices, or higher-dimensional data.
# model. Read sequence of feature vectors and process them with the LSTM layer. Produce a singlee 0 and 1 for each input sequence using the fully connected layer and sigmoid activation function.
class LSTMModel(nn.Module): # nn module is the base class for all neural networks modules in PyTorch.
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dropout_prob=0.2, bidirectional=True): # constructor
        super().__init__() # call the init function from the superclass.
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)

        self.fc_1 = nn.Linear(hidden_size, 256) # fully connected first layer.the second number represents how many neurons
        self.fc_2 = nn.Linear(256, 128)  # fully connected second layer
        self.fc_n = nn.Linear(128, num_classes)
        self.bn_1 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout(dropout_prob)

        # Fully Connected Output Layer
        self.fc_n = nn.Linear(128, num_classes)

        # Non-linear activation
        self.relu = nn.ReLU()

    def forward(self, x): # defines forward pass of neural network.
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # internal state

        # LSTM Layer
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
         # Use the last sequence output for prediction
        last_seq_output = output[:, -1, :]  # Shape: [batch_size, hidden_size]

        # Fully connected layers with Batch Normalization and Dropout
        out = self.fc_1(last_seq_output)
        out = self.relu(out)
        out = self.dropout_1(out)
        
        out = self.fc_2(out)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.dropout_1(out)
        

        # Final Output
        out = self.fc_n(out)  
        return out

'''
------------------------------------------------Train Validate Test------------------------------------------------
'''
def train_model(df):
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


######################### Create, Train, Validate and Test LSTM Model #########################
    num_epochs = 50 # number of epochs
    learning_rate = 0.00001 # learning rate
    input_size = X_train_tensor.size(-1) # number of features
    hidden_size = 300 # number of features in hidden state
    num_layers = 3 # number of stacked lstm layers
    num_classes = 1 # number of output classes 
    model = LSTMModel(num_classes, input_size, hidden_size, num_layers, X_train_tensor.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loader = data.DataLoader(data.TensorDataset(X_train_tensor, y_train_tensor), shuffle=True, batch_size=128)
    early_stopping = EarlyStopping(patience=7, delta=0.001)

    # Training loop
    for epoch in range(num_epochs): # update model's weights (epochs) times. each epoch, it will start with the weights of the previous epoch. the hope is to reduce the amount of loss each time.
        model.train() # set the model to train mode.
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch) # forward pass
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validate for hyperparameter tuning
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_rmse = torch.sqrt(criterion(torch.sigmoid(val_predictions), y_val_tensor)).item()
            val_accuracy = ((torch.sigmoid(val_predictions) > 0.5) == y_val_tensor).float().mean().item()
        
        # Note: RMSE is the Root Mean Square Error. Train measures how well the model fits the data it was trained on. Test RMSE is how well it is expected to perform on unseen data.
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val RMSE: {val_rmse:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step()

        # Call early stopping
        early_stopping(loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    # Calculate Test accuracy and RMSE
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze()
        # Ensure that both predictions and actual values are of the same dimension
        y_test_tensor_squeezed = y_test_tensor.squeeze()

        # Calculate RMSE
        test_rmse = torch.sqrt(criterion(torch.sigmoid(test_predictions), y_test_tensor_squeezed)).item()

         # Calculate accuracy
        test_accuracy = ((torch.sigmoid(test_predictions) > 0.5) == y_test_tensor_squeezed).float().mean().item()

    # Extract the first 100 predicted and actual values
    first_100_predictions = (torch.sigmoid(test_predictions) > 0.5).int().numpy()[:100]
    first_100_actual = y_test_tensor_squeezed.numpy()[:100]

    # Print the RMSE, test accuracy, and first R100 predictions and actual values
    print(f"Test RMSE: {test_rmse:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("First 100 Predictions:", first_100_predictions)
    print("First 100 Actual Values:", first_100_actual)
    
    return model