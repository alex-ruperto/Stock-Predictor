import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import alpacaconfig as config
from random_forest_model import preprocess_data
from random_forest_model import train_random_forest_model

api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)

stock_data = api.get_bars('AAPL', TimeFrame.Hour, start="2017-01-01", end="2023-01-01").df
print(stock_data.columns)
if stock_data.empty:
    raise ValueError("No data fetched from Alpaca.")

print(f"Number of data points: {len(stock_data)}")

# Now proceed with training the model
model = train_random_forest_model(stock_data)
