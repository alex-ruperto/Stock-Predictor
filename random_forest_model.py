import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas_ta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger() # create a logger instance

time_interval = 6.5 # Adjust this to whatever the time interval from raw_data in data_processing is. There are 6.5 hourly interval candles in a single trading day.

# features
def preprocess_data(df):
    logger.info("Data preprocessing started.")
    # Check if the column names are in lower case. If they are, convert them to upper case.
    df.columns = map(str.lower, df.columns)

    # compare next day closing price to current day. convert boolean values to integer values
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int) # df['close] is closing price for each candle. target is to get next close higher than current.
    # Simple Moving Averages 1 and 2
    df['sma1'] = pandas_ta.sma(df['close'], length=50*time_interval)
    df['sma2'] = pandas_ta.sma(df['close'], length=100*time_interval)
    
    # Relative Strength Index (RSI)
    df['rsi'] = pandas_ta.rsi(df['close'], length=14*time_interval)

    # Exponential Moving Averages
    df['ema1'] = pandas_ta.ema(df['close'], length=12*time_interval)
    df['ema2'] = pandas_ta.ema(df['close'], length=26*time_interval)

    # Historical Volatility
    df['volatility'] = pandas_ta.stdev(df['close'], length=14*time_interval)

    # Price Rate of Change
    df['roc'] = pandas_ta.roc(df['close'], length=10*time_interval)

    # Average True Range
    df['atr'] = pandas_ta.atr(df['high'], df['low'], df['close'], length=14*time_interval)

    # Handle NaN values
    df.bfill(inplace=True)  # Backward fill
    df.ffill(inplace=True)  # Forward fill
    
    if df.isnull().values.any():
        logger.warning("Warning: NaN values found after preprocessing")
    
    logger.info("Data preprocessing complete.")
    return df

# model training
def train_random_forest_model(df):
    logger.info("Random Forest Classifier Selected. Beginning model training...")

    # selection of features and target
    X = df.drop('target', axis=1) # remove the target column
    y = df['target'] # only the target column

    # Print the feature names used for training
    feature_names = X.columns.tolist()
    logger.info("Features used for training: %s", feature_names)

    # Split into training and test sets
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data available for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf_classifier = RandomForestClassifier(random_state=42)
    # Define the parameters grid for hyperparameter tuning
    parameters = {
        'n_estimators': [200, 300],
        'max_depth': [10, 30],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4]
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=parameters, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Make predictions with the best estimator
    best_rf_classifier = grid_search.best_estimator_
    predictions = best_rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info('Accuracy of the best model: %s', accuracy)
    logger.info(classification_report(y_test, predictions))

    return best_rf_classifier