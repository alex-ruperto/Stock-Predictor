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
        self.order = None
        self.price = None
        self.stop_loss_triggered = False
        self.take_profit_triggered = False
        self.days_since_rebalance = 0
        self.sell_dates = []
        self.buy_dates = []

    def rebalance(self):
        print("Rebalancing portfolio")
        # TODO 

    def next(self):
        self.days_since_rebalance += 1 # add one to the count
        
        if self.days_since_rebalance >= self.p.rebalance_days: # execute rebalance and reset counter variable
            self.rebalance()
            self.days_since_rebalance = 0 

        if not self.position or self.order: # you have no position in the market. no active trades either.
            return
        
        price_change = (self.data.close[0] - self.price) / self.price # calculate % change in price since purchase

        # check for when price_change crosses a stop-loss or take-profit tier and sell accordingly.

        # Tier 3
        if price_change <= -self.p.stop_loss_3 and self.position.size > 0: 
            self.close() 
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold
            print(f"Tier 3 Stop loss triggered! Selling all positions on {bt.num2date(self.data.datetime[0])}")
            return
        if price_change >= self.p.take_profit_3 and self.position.size > 0:
            self.close()
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold
            print(f"Tier 3 Take-Profit triggered! Selling all positions on {bt.num2date(self.data.datetime[0])}")
            return
        # Tier 2
        if price_change <= -self.p.stop_loss_2 and self.position.size > 0:  
            self.sell(size = min(self.position.size * 0.4, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold
            print(f"Tier 2 Stop loss triggered! Selling 40% on {bt.num2date(self.data.datetime[0])}")
            return
        if price_change >= self.p.take_profit_2 and self.position.size > 0:
            self.sell(size = min(self.position.size * 0.4, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold
            print(f"Tier 2 Take-Profit triggered! Selling 40% on {bt.num2date(self.data.datetime[0])}")
            return

        # Tier 1
        if price_change <= -self.p.stop_loss_1 and self.position.size > 0: 
            self.sell(size = min(self.position.size * 0.2, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold
            print(f"Tier 1 Stop loss triggered! Selling 20% on {bt.num2date(self.data.datetime[0])}")
            return
        if price_change >= self.p.take_profit_1 and self.position.size > 0:
            self.sell(size = min(self.position.size * 0.2, self.position.size))
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold
            print(f"Tier 1 Take-Profit triggered! Selling 20% on {bt.num2date(self.data.datetime[0])}")
            return


        
