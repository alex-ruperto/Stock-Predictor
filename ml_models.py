import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.impute import SimpleImputer  # Add this import

# features
def preprocess_data(df):
    # compare next day closing price to current day. convert boolean values to integer values
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) 
    df['SMA1'] = df['Close'].rolling(window=50).mean()
    df['SMA2'] = df['Close'].rolling(window=200).mean()
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
    """Compute Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def stochastic_oscillator(data, window=14):
    """Compute Stochastic Oscillator"""
    low_min = data.rolling(window=window).min()
    high_max = data.rolling(window=window).max()
    k_line = 100 * ((data - low_min) / (high_max - low_min))
    d_line = k_line.rolling(window=3).mean()
    return k_line, d_line

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

    # Define LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train_reshaped, y_train_reshaped, epochs=30, validation_data=(X_test_reshaped, y_test_reshaped))

    # Predictions
    y_pred = model.predict(X_test_reshaped)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test_reshaped, y_pred)
    
    print(f"Model Accuracy on Test Set with LSTM: {accuracy:.2f}")
    
    return model

