import pandas as pd
import pandas_ta as ta
from Preprocessors.base_preprocessor import DataPreprocessor
from Utils.logger_config import configure_logger

class RFCPreprocessor(DataPreprocessor):
    def __init__(self):
        self.logger = configure_logger(self.__class__.__name__) # create new logger with the class name as a string. 'RFCPreprocessor'

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        time_interval = 6.5 # Adjust this to whatever the time interval from raw_data in data_processing is. There are 6.5 hourly interval candles in a single trading day.
        self.logger.info("Data preprocessing started.")
        df.columns = map(str.lower, df.columns)

        # compare next day closing price to current day. convert boolean values to integer values
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int) # df['close] is closing price for each candle. target is to get next close higher than current.
        # Simple Moving Averages 1 and 2
        df['sma1'] = ta.sma(df['close'], length=50*time_interval)
        df['sma2'] = ta.sma(df['close'], length=100*time_interval)
        
        # Relative Strength Index (RSI)
        df['rsi'] = ta.rsi(df['close'], length=14*time_interval)

        # Exponential Moving Averages
        df['ema1'] = ta.ema(df['close'], length=12*time_interval)
        df['ema2'] = ta.ema(df['close'], length=26*time_interval)

        # Historical Volatility
        df['volatility'] = ta.stdev(df['close'], length=14*time_interval)

        # Price Rate of Change
        df['roc'] = ta.roc(df['close'], length=10*time_interval)

        # Average True Range
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14*time_interval)

        # Handle NaN values
        df.bfill(inplace=True)  # Backward fill
        df.ffill(inplace=True)  # Forward fill
        
        if df.isnull().values.any():
            self.logger.warning("Warning: NaN values found after preprocessing")
        
        self.logger.info("Data preprocessing complete.")
        return df