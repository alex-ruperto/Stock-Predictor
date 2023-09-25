from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# df stands for dataframe. dataframe is a data structure that is 2D like a spreadsheet.
def train_model(df): # pandas dataframe expected
    # handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df_imputed = df.copy()
    df_imputed[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line']] = imputer.fit_transform(df[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line']])
    
    # split data into training and test sets:
    X = df_imputed[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line']]
    y = df_imputed['Target']


    # 80% of data for training, X_train, y_train
    # 20 % of data for testing (X_test, y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    # Explanation of this here: https://builtin.com/data-science/random-forest-python-deep-dive
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_prediction = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    
    print(f"Model Accuracy on Test Set: {accuracy:.2f}")
    
    return clf

def preprocess_data(df):
    df['SMA1'] = df['Close'].rolling(window=50).mean()
    df['SMA2'] = df['Close'].rolling(window=200).mean()
    # compare next day closing price to current day. convert boolean values to integer values
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) 
    df['RSI'] = rsi_formula(df['Close'], window=14)
    df['MACD_Line'], df['Signal_Line'] = macd_formula(df['Close'], short_window=12, long_window=26, signal_window=9)
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