import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas_ta
from sklearn.preprocessing import StandardScaler
time_interval = 6.5 # Adjust this to whatever the time interval from raw_data in data_processing is. There are 6.5 hourly interval candles in a single trading day.

# features
def preprocess_data(df):
    # Check if the column names are in lower case. If they are, convert them to upper case.
    df.columns = map(str.capitalize, df.columns)

    # compare next day closing price to current day. convert boolean values to integer values
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # df['Close] is closing price for each candle. target is to get next Close higher than current.
    # Simple Moving Averages 1 and 2
    df['SMA1'] = pandas_ta.sma(df['Close'], length=50*time_interval)
    df['SMA2'] = pandas_ta.sma(df['Close'], length=100*time_interval)
    
    # Relative Strength Index (RSI)
    df['RSI'] = pandas_ta.rsi(df['Close'], length=14*time_interval)

    # Exponential Moving Averages
    df['EMA1'] = pandas_ta.ema(df['Close'], length=12*time_interval)
    df['EMA2'] = pandas_ta.ema(df['Close'], length=26*time_interval)

    # Historical Volatility
    df['Volatility'] = pandas_ta.stdev(df['Close'], length=14*time_interval)

    # Price Rate of Change
    df['ROC'] = pandas_ta.roc(df['Close'], length=10*time_interval)

    # Average True Range
    df['ATR'] = pandas_ta.atr(df['High'], df['Low'], df['Close'], length=14*time_interval)

    # Fill any NaN values with the mean
    df = df.fillna(df.mean())

    return df

# end of features

def train_random_forest_model(df):
    df_processed = preprocess_data(df)

    # selection of features and target
    X = df_processed.drop('Target', axis=1) # remove the target column
    y = df_processed['Target'] # only the target column

    # split into training and test sets
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data available for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose = 1)
    rf_classifier.fit(X_train, y_train)

    # make predictions and evaluate the model.
    predictions = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, predictions))

    return rf_classifier