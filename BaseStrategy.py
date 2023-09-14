import backtrader as bt

class BaseStrategy(bt.Strategy): # base strategy class that implements take-profit and stop-loss.
    params = (
        ('stop_loss_1', 0.15),  # 15% stop loss
        ('take_profit_1', 0.10), # 10% take profit
        ('stop_loss_2', 0.30), # 30% stop loss 
        ('take_profit_2', 0.25), # 25% take_profit
        ('stop_loss_3', 0.50), # 50% stop loss
        ('take_profit_3', 0.50), # 50% take_profit
        ('rebalance_days', 126) # number of trading days before performing a rebalance on the portfolio
    )

    def __init__(self):
        self.current_index = 0
        self.cash_values = []
        self.account_values = []
        self.position_sizes = []
        self.sell_dates = []
        self.buy_dates = []
        self.order = None
        self.price = None
        self.stop_loss_triggered = False
        self.take_profit_triggered = False
        self.days_since_rebalance = 0
        
    def rebalance(self):
        print("Rebalancing portfolio")
        # TODO 

    def update_lists(self): # for the daily changing lists
        # Update the values at the current index
        self.position_sizes.append(self.position.size)
        self.cash_values.append(self.broker.get_cash())
        self.account_values.append(self.broker.get_value())

    def next(self):
        # print("Processing date:", bt.num2date(self.data.datetime[0]))
        self.days_since_rebalance += 1 # add one to the count
        
        if self.days_since_rebalance >= self.p.rebalance_days: # execute rebalance and reset counter variable
            self.rebalance()
            self.days_since_rebalance = 0 
        
        if self.price is not None:
            price_change = (self.data.close[0] - self.price) / self.price # calculate % change in price since purchase

        if not self.position:
            return
        # check for when price_change crosses a stop-loss or take-profit tier and sell accordingly.
        # Tier 3
        elif price_change <= -self.p.stop_loss_3 and self.position.size > 0:
            self.close()
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
            print(f'BaseStrategy Sell - Date: {bt.num2date(self.data.datetime[0])}, Price Change: {price_change:.2f}%')
        elif price_change >= self.p.take_profit_3 and self.position.size > 0:
            self.close()
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
            print(f'BaseStrategy Sell - Date: {bt.num2date(self.data.datetime[0])}, Price Change: {price_change:.2f}%')
        
        # Tier 2
        elif price_change <= -self.p.stop_loss_2 and self.position.size > 0:
            self.sell(size=min(self.position.size * 0.4, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
            print(f'BaseStrategy Sell - Date: {bt.num2date(self.data.datetime[0])}, Price Change: {price_change:.2f}%')
        elif price_change >= self.p.take_profit_2 and self.position.size > 0:
            self.sell(size=min(self.position.size * 0.4, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
            print(f'BaseStrategy Sell - Date: {bt.num2date(self.data.datetime[0])}, Price Change: {price_change:.2f}%')
        
        # Tier 1
        elif price_change <= -self.p.stop_loss_1 and self.position.size > 0:
            self.sell(size=min(self.position.size * 0.2, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
            print(f'BaseStrategy Sell - Date: {bt.num2date(self.data.datetime[0])}, Price Change: {price_change:.2f}%')
        elif price_change >= self.p.take_profit_1 and self.position.size > 0:
            self.sell(size=min(self.position.size * 0.2, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
            print(f'BaseStrategy Sell - Date: {bt.num2date(self.data.datetime[0])}, Price Change: {price_change:.2f}%')

        self.update_lists()