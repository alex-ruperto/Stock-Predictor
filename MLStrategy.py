import backtrader as bt
from BaseStrategy import BaseStrategy

class MLStrategy (BaseStrategy):
    params = (
        ("sma1_period", 50),
        ("sma2_period", 200),
        ("rsi_period", 14),
        ("macd1_period", 12),
        ("macd2_period", 26),
        ("macd_signal_period", 9),
        ("bb_period", 20)
    )
    
    def __init__(self):
        super().__init__()
        self.initialize_lists(self.params.sma2_period)
        # Simple Moving Average
        self.sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period = self.params.sma1_period)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period = self.params.sma2_period)

        # RSI
        self.rsi = bt.indicators.RelativeStrengthIndex(period = self.params.rsi_period)

        # MACD
        self.macd = bt.indicators.MACD(self.data.close, period_me1 = self.params.macd1_period, 
                                       period_me2 = self.params.macd2_period, period_signal = self.params.macd_signal_period)
        
        # Bollginer Bands
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period = self.params.bb_period)

    def next(self):
        if self.sma1 > self.sma2 and self.rsi < 30:
            self.buy()
            self.buy_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
        elif self.sma1 < self.sma2 or self.rsi > 70:
            if self.position.size > 0:
                self.sell()
                self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold

        self.update_lists()

