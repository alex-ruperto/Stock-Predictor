from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
    
    return clf