import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import alpacaconfig as config
from random_forest_model import preprocess_data
from random_forest_model import train_random_forest_model

api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)

stock_data = api.get_bars('AAPL', TimeFrame.Day, start="2020-01-01", end="2023-01-01").df
print("Shape of stock_data:", stock_data.shape)
if stock_data.empty:
    raise ValueError("No data fetched from Alpaca.")

df = preprocess_data(stock_data)
print("Shape of df after preprocessing:", df.shape)
if df.empty:
    raise ValueError("DataFrame is empty after preprocessing.")


# Now, check for NaN values again
print("NaN counts after trimming:", df.isna().sum())

# Now proceed with training the model
model = train_random_forest_model(df)
