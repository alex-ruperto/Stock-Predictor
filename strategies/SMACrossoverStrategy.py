import backtrader as bt
from Strategies.BaseStrategy import BaseStrategy

class SMACrossoverStrategy(BaseStrategy):  # class definition with BaseStrategy as the parent
    # these two periods will refer to how many days the moving average will be calculated.
    # one moving average for a short period and one for a long period.
    params = dict(
        short_period=50,
        long_period=200
    )

    def __init__(self):  # constructor for when the SMACrossover class is called.
        # create short simple moving average using short_period and long simple moving average using long_period
        # parameters. The crossover indicator is checking for crossovers between two data lines.
        # + 1 for upward, -1 for downward crossover
        super().__init__()
        self.initialize_lists(self.p.long_period)
        self.sma_short = bt.ind.SMA(self.data.close, period=self.p.short_period)  # initialize sma_short as a bt SMA indicator using the short period
        self.sma_long = bt.ind.SMA(self.data.close, period=self.p.long_period)  # initialize sma_long as a bt SMA indicator using the long period
        self.crossover = bt.ind.CrossOver(self.sma_short, self.sma_long)  # use crossover indicator with sma_short and sma_long
        self.crossover_history = [] # record the history of the crossover into a list
        self.pending_order = None # tracks the buy/sell decision
        self.in_golden_cross = False # boolean to check whether it is in a golden cross or not
        self.in_death_cross = False # boolean to check whether it is in a death cross or not
        

    # Golden Cross. Short SMA Crosses over Long SMA
    def should_buy(self):
        return self.sma_short[0] > self.sma_long[0] and not self.in_golden_cross
    
    # Death Cross. Short SMA Crosses under Long SMA
    def should_sell(self):
        return self.sma_short[0] < self.sma_long[0] and not self.in_death_cross


    # call next method for each bar/candle in backtest.
    def next(self):
        super().next()

        # Add the crossover to the history first
        self.crossover_history.append(self.crossover[0]) # append self.crossover to the crossover_history list
        current_date = bt.num2date(self.data.datetime[0])

        
        # update the states
        if self.should_buy(): 
            self.in_golden_cross = True
            self.in_death_cross = False
            self.pending_order = 'buy'
        
        elif self.should_sell():
            self.in_golden_cross = False
            self.in_death_cross = True
            self.pending_order = 'sell'
        
        # Add a variable window to confirm trend
        confirmation_days = 3 # !!!keep in mind, execution will take an extra day!!!
        consistent_crossover = all(x == self.crossover[0] for x in self.crossover_history[-confirmation_days:])
        # Check the last confirmation_days - 1 days since we've already added today's value\
        # - in confirmation_days is used to count from the end of the list backwards. confirm each item in
        # this list is ('x') to the current crossover value. it will return true or false. the all return
        # true if all the items in the iterable return true. otherwise, false.
        # colon indicates a slice from starting point to the end.

         # take 10% of your cash and then perform floor division by the closing price of the stock.
        size_to_buy = (self.broker.get_cash() * 0.1) // self.data.close[0]  
        if size_to_buy < 1:  # if the size_to_buy value is less than one
            size_to_buy = 1  # buy one

        if self.pending_order == 'buy' and consistent_crossover: # if the pending order is a buy and consistent_crossover
            self.buy(size=size_to_buy) # 
            self.price = self.data.close[0]
            self.buy_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it bought
            self.pending_order = None # reset pending order
            

        elif self.pending_order == 'sell' and consistent_crossover and self.position.size > 0: 
            # if the pending_order is a sell, consistent and there are shares in the account, 
            positions_to_sell = self.position.size // 2 # sell half of all shares
            self.sell(size=positions_to_sell) # sell
            self.sell_dates.append(bt.num2date(self.data.datetime[0])) # add the date of when it sold
            self.pending_order = None # reset pending order
            
        self.update_lists()

        

        
        