import backtrader as bt
import datetime
from datetime import timedelta
from datetime import date

class BaseStrategy(bt.Strategy): # base strategy class that implements take-profit and stop-loss.
    params = (
        ('stop_loss_1', 0.15),  # 15% stop loss
        ('take_profit_1', 0.10), # 10% take profit
        ('stop_loss_2', 0.30), # 30% stop loss 
        ('take_profit_2', 0.25), # 25% take_profit
        ('stop_loss_3', 0.50), # 50% stop loss
        ('take_profit_3', 0.50), # 50% take_profit
        ('rebalance_days', 126), # number of trading days before performing a rebalance on the portfolio
        ('weekly_cash_injection', 100.0) # amount of cash to inject
    )

    def __init__(self):
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
        self.just_rebalanced = False
        self.current_index = 0
        self.dates = [bt.num2date(x) for x in self.data.datetime.array]
        if self.dates:
            self.start_date = self.dates[0]
            self.end_date = self.dates[-1]
        else:
            self.start_date = None
            self.end_date = None
        self.last_injection_date = None
        self.days_since_last_injection = 7


    def add_cash(self):
        current_date = self.data.datetime.date(0)
        
        # If it's the first time or a week has passed since the last injection
        days_since_last_injection = (current_date - self.last_injection_date).days if self.last_injection_date else 7
        if days_since_last_injection >= 7:
            self.broker.add_cash(self.p.weekly_cash_injection)
            self.last_injection_date = current_date

    def rebalance(self):
        # Calculate total portfolio value
        total_value = self.broker.get_value()

        # Calculate desired cash
        desired_cash_value = total_value * 0.5

        # Calculate current cash
        current_cash_value = self.broker.get_cash()

        # Calculate the difference between desired and current values
        cash_difference = desired_cash_value - current_cash_value

        # Buy or sell stock to achieve the desired allocation
        if cash_difference < 0:  # Buy stock
            size_to_buy = -cash_difference // self.data.close[0]
            if size_to_buy < 1:
                size_to_buy = 1
            self.buy(size=size_to_buy)
            self.price = self.data.close[0]
            self.buy_dates.append(bt.num2date(self.data.datetime[0]))
            print(f"Rebalanced by buying {size_to_buy} shares")

        elif cash_difference < 0:  # Sell stock
            size_to_sell = abs(cash_difference) // self.data.close[0]

            # Ensure you don't sell more than you own
            if size_to_sell > self.position.size:
                size_to_sell = self.position.size

            # Only sell if you have shares to sell
            if size_to_sell > 0:
                self.sell(size=size_to_sell)
                self.sell_dates.append(bt.num2date(self.data.datetime[0]))
                print(f"Rebalanced by selling {size_to_sell} shares")



    def initialize_lists(self):
        self.position_sizes = [self.position.size]
        self.cash_values = [self.broker.get_cash()]
        self.account_values = [self.broker.get_value()]
    
    def update_lists(self):
        self.cash_values.append(self.broker.get_cash())  # append the current cash value to the list.
        self.account_values.append(self.broker.get_value()) # append the current account value list
        self.position_sizes.append(self.position.size) # append the current position size to list
    
    def take_profit_and_stop_loss(self):
        price_change = 0.0
        if self.price is not None:
            price_change = (self.data.close[0] - self.price) / self.price # calculate % change in price since purchase
        if self.position.size == 0:
            return
        
        # check for when price_change crosses a stop-loss or take-profit tier and sell accordingly.
        # Tier 3
        elif price_change <= -self.p.stop_loss_3 and self.position.size > 0:
            self.close()
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
        elif price_change >= self.p.take_profit_3 and self.position.size > 0:
            self.close()
            self.sell_dates.append(bt.num2date(self.data.datetime[0]))
        
        # Tier 2
        elif price_change <= -self.p.stop_loss_2 and self.position.size > 0:
            if self.position.size <= 1 and self.position.size >= 0:
                self.close()
            else:
                # Round the sell order size to the nearest whole number
                size_to_sell = round(min(self.position.size * 0.4, self.position.size))
                self.sell(size=size_to_sell)
                self.sell_dates.append(bt.num2date(self.data.datetime[0]))
        elif price_change >= self.p.take_profit_2 and self.position.size > 0:
            if self.position.size == 1 and self.position.size > 0:
                self.close()
            else:
                # Round the sell order size to the nearest whole number
                size_to_sell = round(min(self.position.size * 0.4, self.position.size))
                self.sell(size=size_to_sell)
                self.sell_dates.append(bt.num2date(self.data.datetime[0]))

        # Tier 1
        elif price_change <= -self.p.stop_loss_1 and self.position.size > 0:
            if self.position.size == 1 and self.position.size >= 0:
                self.close()
            else:
                # Round the sell order size to the nearest whole number
                size_to_sell = round(min(self.position.size * 0.2, self.position.size))
                self.sell(size=size_to_sell)
                self.sell_dates.append(bt.num2date(self.data.datetime[0]))

        elif price_change >= self.p.take_profit_1 and self.position.size > 0:
            if self.position.size == 1 and self.position.size >= 0:
                self.close()
            else:
                # Round the sell order size to the nearest whole number
                size_to_sell = round(min(self.position.size * 0.2, self.position.size))
                self.sell(size=size_to_sell)
                self.sell_dates.append(bt.num2date(self.data.datetime[0]))

            

    def next(self):
        self.days_since_rebalance += 1 # add one to the count
        
        if self.days_since_rebalance >= self.p.rebalance_days: # execute rebalance and reset counter variable
            self.rebalance()
            self.days_since_rebalance = 0
            self.just_rebalanced = True
        
        if not self.just_rebalanced:
            self.take_profit_and_stop_loss()
        
        self.just_rebalanced = False
        
        
        