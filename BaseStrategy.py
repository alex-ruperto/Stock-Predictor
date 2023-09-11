import backtrader as bt

class BaseStrategy(bt.Strategy): # base strategy class that implements take-profit and stop-loss.
    params = (
        ('stop_loss', 0.01),  # 10% stop loss
        ('take_profit', 0.01) # 10% take profit
    )

    def __init__(self):
        self.order = None
        self.price = None
        self.stop_loss_triggered = False
        self.take_profit_triggered = False

    def should_buy(self):
        # placeholder method to be overridden by a child strategy class.
        return False
    
    def should_sell(self):
        return False

    def next(self):
        if not self.position: # you have no position in the market. no active trades either.
            return
        
        if self.order:
            return
        
        if self.should_buy():
            self.buy() # should_buy is meant to be overriden by child class.
            self.price = self.data.close[0]
        elif self.position and self.should_sell():
            self.sell() # should_sell is meant to be overriden by child class.
            self.price = None
        
        # if the closing price is below the calculated stop-loss level
        if self.data.close[0] < (1 - self.p.stop_loss) * self.price:
            self.stop_loss_triggered = True

        # if the closing price is above the calculated stop-loss level
        if self.data.close[0] > (1 + self.p.take_profit) * self.price:
            self.take_profit_triggered = True

        if self.stop_loss_triggered:
            self.close()
            self.stop_loss_triggered = False # reset variable
            self.price = None
        
        if self.take_profit_triggered:
            self.close()
            self.take_profit_triggered = False # reset variable
            self.price = None
        

