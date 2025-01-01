import alpaca_trade_api as tradeapi
import alpacaconfig as config
import pandas as pd
from alpaca_trade_api import TimeFrame


def fetch_stock_data(selected_ticker: str, start_date: str, end_date: str, timeframe:str) -> pd.DataFrame:
    """
        Fetch stock data for a given ticker from Alpaca.

        Args:
            selected_ticker (str): Stock ticker symbol (e.g., 'AAPL').
            start_date (str): Start date for fetching data (YYYY-MM-DD).
            end_date (str): End date for fetching data (YYYY-MM-DD).
            timeframe (str): Timeframe for stock data. Acceptable values:
            [
                "Min", Minute,
                "Hour", Hour,
                "Day", Day,
                "Week", Week,
                "Month", Month,
            ]

        Returns:
            pd.DataFrame: Stock data with datetime index.
    """

    if timeframe == 'Min':
        timeframe = TimeFrame.Min
    elif timeframe == 'Hour':
        timeframe = TimeFrame.Hour
    elif timeframe == 'Day':
        timeframe = TimeFrame.Day
    elif timeframe == 'Week':
        timeframe = TimeFrame.Week
    elif timeframe == 'Month':
        timeframe = TimeFrame.Month
    else:
        raise ValueError('Timeframe must be one of Min, Hour, Day, Week, Month')

    alpaca_api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL)
    stock_data = alpaca_api.get_bars(selected_ticker, timeframe, start_date, end=end_date).df
    if stock_data.empty:
        raise ValueError("No data fetched from Alpaca.")

    return stock_data