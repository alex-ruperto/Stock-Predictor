import numpy as np
import pandas as pd
from sklearn.ensemble import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# df stands for dataframe. dataframe is a data structure that is 2D like a spreadsheet.
def train_model(df):
    # handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df_imputed = df.copy()
    df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed), columns=df.columns)

    # Normalize Data
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(df_imputed.values)

    # create sequences
    X, y = [], []
    time_step = 60
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i+time_step])
        y.append(scaled_data[i+ time_step, df.columns.get_loc('Target')])
    
    X,y = np.array(X), np.array(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

    return model

# features
def preprocess_data(df):
    df['SMA1'] = df['Close'].rolling(window=50).mean()
    df['SMA2'] = df['Close'].rolling(window=200).mean()
    # compare next day closing price to current day. convert boolean values to integer values
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) 
    df['RSI'] = rsi_formula(df['Close'], window=14)
    df['MACD_Line'], df['Signal_Line'] = macd_formula(df['Close'], short_window=12, long_window=26, signal_window=9)
    df['Upper_Bollinger'], df['Lower_Bollinger'] = bollinger_bands(df['Close'])
    df['K_Line'], df['D_Line'] = stochastic_oscillator(df['Close'])

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

def bollinger_bands(data, window=20, num_std_dev=2):
    """Computer Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def stochastic_oscillator(data, window=14):
    """Computer Stochastic Oscillator"""
    low_min = data.rolling(window=window).min()
    high_max = data.rolling(window=window).max()
    k_line = 100 * ((data - low_min) / (high_max - low_min))
    d_line = k_line.rolling(window=3).mean()
    return k_line, d_line