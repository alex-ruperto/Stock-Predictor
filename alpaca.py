import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import alpacaconfig as config
from lstm_model import preprocess_data
from lstm_model import train_model

api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)

stock_data = api.get_bars('AAPL', TimeFrame.Day, start="2020-01-01", end="2023-01-01").df

df = preprocess_data(stock_data)

model = train_model(df)
