from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# df stands for dataframe. dataframe is a data structure that is 2D like a spreadsheet.
def train_model(df): # pandas dataframe expected
    # handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df_imputed = df.copy()
    df_imputed[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 
                'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 
                'D_Line']] = imputer.fit_transform(df[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 
                'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line']])
    
    # split data into training and test sets:
    X = df_imputed[['SMA1', 'SMA2', 'RSI', 'MACD_Line', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger', 'K_Line', 'D_Line']]
    y = df_imputed['Target']


    # 80% of data for training, X_train, y_train
    # 20 % of data for testing (X_test, y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Parameter grid to search
    parameter_grid = {
        'n_estimators': [10, 50, 200], # this is the number of trees in the forest
        'max_depth':[None, 10, 20], # Maximum depth of each tree
        'min_samples_split':[2, 5], # minimum number of samples to split an node. if samples < this, become leaf node
        'min_samples_leaf':[1, 2], # minimum number of samples required to be at a leaf node
        'max_features':[None] # determines maximum number of features considered
    }
    # number of candidates = multiplication of the size of each hyperparameter. 
    # number of fits (model training runs) = number of candidates * number of folds (cv)

    # Train model
    # Explanation of this here: https://builtin.com/data-science/random-forest-python-deep-dive
    clf = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=clf, param_grid=parameter_grid, cv=3, 
                               n_jobs=-1, verbose=1, scoring='accuracy', error_score='raise')
    # cv is how many partitions the data should be split into. train on 2, validate on 3.
    # n_jobs=-1 means all available processors will be used
    # verbose=1 function will print detailed info on progress of the grid search.
    # scoring=accuracy model with the highest accuracy on the validation set will be considered the best.

    grid_search.fit(X_train, y_train)
    # get best parameters
    best_parameters = grid_search.best_params_

    # train the model using the best parameters
    best_clf = grid_search.best_estimator_

    y_prediction = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    
    print(f"Model Accuracy on Test Set with best parameters: {accuracy:.2f}")
    
    return best_clf

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