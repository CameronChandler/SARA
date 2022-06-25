import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates
from yahooquery import Ticker


SMALL, MED, LARGE, LW = 18, 24, 30, 3
plt.rc('axes', titlesize=MED)    # fontsize of the axes title
plt.rc('axes', labelsize=MED)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL) # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL) # fontsize of the tick labels
plt.rc('legend', fontsize=MED)   # legend fontsize
plt.rc('font', size=LARGE)       # controls default text sizes

TRADING_DAYS = 252
INITIAL_CASH = 20_000
TOMORROW = dt.datetime.today().date() + dt.timedelta(days=1)
COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', '#ffe119', '#46f0f0', '#008080', 
          '#9a6324', '#808000', '#000075', '#aaffc3', '#000000', '#808080']

TEST, PROD = 0, 1
SIIF = 1

class CashFlow:
    ''' Holds the meta data of one cash_flow '''
    
    def __init__(self, row):
        self.cash = row.price
        self.date = row.date
        
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return f'{self.date}: ${self.cash}'
    
class Dividend(CashFlow):
    ''' Holds the meta data of one dividend payment '''
    
    def __init__(self, row):
        super().__init__(row)
    
class Share:
    ''' Holds the meta data of one share in a composition 
    timeline: dataframe of share information'''
    
    def __init__(self, transactions, dates):
        self.code = transactions.iloc[0].code
        self.transactions = transactions.sort_values('date')
        self.dates = dates
        self.timeline = self._gen_timeline()
            
    def _gen_timeline(self):
        # Initialise empty timeline
        timeline = pd.DataFrame({'units': 0, 'transaction_price': 0, 'fees': 0}, index=self.dates)
        
        timeline = self._add_transactions_to_timeline(timeline)
        
        timeline = self._add_prices_to_timeline(timeline)
        
        return timeline.fillna(0)
            
    def _add_transactions_to_timeline(self, timeline):
        for _, row in self.transactions.iterrows():
            timeline.loc[row.date:, 'units'] += row.units
            timeline.loc[row.date, 'fees'] = row.fee
            timeline.loc[row.date, 'transaction_price'] = row.price
            
        return timeline
    
    def _add_prices_to_timeline(self, timeline, test=False):
        
        prices = self._gen_real_prices()
        
        if prices is None:
            prices = self._gen_fake_prices()
        
        if test:
            return(timeline, prices)
        timeline = pd.merge(timeline, prices, on='date', how='left').ffill()
            
        return timeline
            
    def _gen_real_prices(self):
        ''' Attempt to generate real prices. Returns None if prices cannot be found '''
        prices = Ticker(f'{self.code}.AX').history(start=self.dates[0], end=TOMORROW)
        if isinstance(prices, dict):
            return None
        
        prices = prices.reset_index()[['date', 'close']].set_index('date')
        prices.columns = ['price']
        return prices
    
    def _gen_fake_prices(self):
        ''' Use linear interpolation to estimate prices '''
        # Initialise prices df
        prices = pd.DataFrame({'price': 0}, index=self.dates)

        # Between each pair of transactions, calculate the slope to perform linear interpolation
        prev = self.transactions.iloc[0]
        for _, curr in self.transactions.iloc[1:].iterrows():
            prices.loc[curr.date] = curr.price
            slope = (curr.price - prev.price) / (len(prices.loc[prev.date: curr.date]) - 1)
            prices.loc[prev.date: curr.date] = slope
            prices.loc[prev.date] = prev.price

            prices.loc[prev.date: curr.date] = prices.loc[prev.date: curr.date].cumsum()
            
        return prices

    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return f'{self.code}'

class Comp:
    ''' Represents a portfolio composition 
    name:            name of composition (shows up on graphs)
    filename:        filename for CSV file containing composition meta data
    shares:          list of Share objects
    curr_shares:     list of Share objects that are currently owned
    cash_flows:      list of CashFlow objects
    dividends:       list of Dividend objects
    portfolio_value: numpy array of net portfolio value over time
    returns:         portfolio performance'''
    
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.shares, self.cash_flows, self.dividends = self.load_comp()
        self.curr_shares = [share for share in self.shares if share.timeline.iloc[-1]['units'] != 0]
        self.portfolio_value = self.get_portfolio_value()
        self.returns = self.get_returns()
        
    def load_comp(self):
        ''' Loads units, price, sell_price, buy_date, sell_date and code as registered in the data csv '''
        # First, work out when the portfolio starts
        data = pd.read_csv('./comps/'+self.filename+'.csv').dropna(how='all')
        data['date'] = pd.to_datetime(data.date, dayfirst=True).dt.date
        data['price'] = data.price.astype(float)
        min_date = data.date.min()
                
        self.dates = Ticker('NDQ.AX').history(start=min_date, end=TOMORROW).reset_index()['date']

        # Calculate the net units owned of each share, and split data into past and current shares
        share_data = data[~data.code.isin(['CASH_FLOW', 'DIVIDEND'])]
        share_data['units'] = share_data.units.astype(int)
        share_data['fee']   = share_data.fee.astype(float)
        
        shares = []
        cash_flows = []
        dividends = []
        for code in share_data.code.unique():
            shares.append(Share(share_data[share_data.code == code], self.dates))
            
        cash_flow_data = data[data.code == 'CASH_FLOW']
        dividends_data = data[data.code == 'DIVIDEND']
        
        for _, row in cash_flow_data.iterrows():
            cash_flows.append(CashFlow(row))
        
        for _, row in dividends_data.iterrows():
            dividends.append(Dividend(row))
                
        return shares, cash_flows, dividends
    
    def get_portfolio_value(self, end_of_day=False):
        ''' Calculate the portfolio value for each day '''
        total_share_value = pd.Series(0, index=self.shares[0].timeline.index)
        for share in self.shares:
            total_share_value += share.timeline['price'] * share.timeline['units']
            
        return total_share_value + self.get_cash(end_of_day)
    
    def get_cash(self, end_of_day):        
        ''' Return a numpy array representing the amount of cash in the portfolio at each day '''
        # Initialise starting balance
        # cash = [20_000, 20_000, 20_000, 20_000] 1xD 
        cash = pd.Series(self.cash_flows[0].cash, index=self.dates)

        for cash_event in self.cash_flows[1:] + self.dividends:
            # When should cash_flow count
            if end_of_day:
                effective_period = np.where(self.dates >  cash_event.date , 1, 0)
            else:
                effective_period = np.where(self.dates >= cash_event.date , 1, 0)

            cash += cash_event.cash * effective_period

        for share in self.shares:
            # Add units_diff column
            share.timeline['units_diff'] = share.timeline['units'].diff().fillna(share.timeline['units'])
            
            for date, row in share.timeline[share.timeline.units_diff != 0].iterrows():
                # Change cash 'from then onwards'
                cash.loc[date:] += -row['units_diff']*row['transaction_price'] - row['fees']

        return cash
    
    def get_returns(self):
        ''' Calculate portfolio returns at each day. Accounts for cash injection/withdrawal
        Method from https://www.fool.com/about/how-to-calculate-investment-returns/
        
        Data looks like:
        date:                       date
        end_of_day_portfolio_value: units @ price (before cash flow added)
        cash_flow:                  cash_flow added at end of day
        value_after_cash_flow:      sum above two
        last_base:                  end_of_day_portfolio at beginning of current holding period
        cum_HPR:                    cumulative return for current holding period only
        HPR:                        1 if not end of current HP, if at end, represents return for that HP
        cum_return:                 cumulative total HPRs
        return:                     final calculated return on investment at any given time
        '''
        data = pd.DataFrame(self.get_portfolio_value(end_of_day=True), 
            columns=['end_of_day_portfolio_value'],
            index=self.portfolio_value.index)

        data['cash_flow'] = 0
        for cash_flow in self.cash_flows[1:]:
            data.loc[cash_flow.date, 'cash_flow'] = cash_flow.cash

        data['value_after_cash_flow'] = data['end_of_day_portfolio_value'] + data['cash_flow']
        data['last_base'] = np.where(data['cash_flow'] != 0, data['value_after_cash_flow'], np.nan)
        data['last_base'][0] = data['end_of_day_portfolio_value'][0]
        data['last_base'] = data['last_base'].ffill()

        # Holding Period Return
        data['cum_HPR'] =  data['end_of_day_portfolio_value'] / data['last_base'].shift()
        data['cum_HPR'][0] = 1
        data['HPR'] = np.where(data['cash_flow'] != 0, data['cum_HPR'], 1)
        data['cum_return'] = np.cumprod(data['HPR'])
        data['return'] = np.where(data['cash_flow'] == 0,
                                  data['cum_HPR']*data['cum_return'], data['cum_return']) 

        return data['return']

    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return f'{self.name}: {self.curr_shares}'

################################ Plotting functions ################################

def plot_comps(comps, filename='strategy_comparison', save=False, scale=1):
    ''' Plots the performance of each comp '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

    rs = comps[0].returns
    start = rs[rs.diff() != 0].index[1]

    for comp in comps:
        x = rs[start:]
        y = comp.returns[start:]
        ax.plot(x[~x.index.duplicated(keep='first')].index, y[~y.index.duplicated(keep='first')], label=comp.name, alpha=0.9, lw=LW)

    # Plot Bank
    rate = 0.03
    cum_balance = list(np.cumprod((1+rate)**(1/TRADING_DAYS) * np.ones(len(rs[start:]))))
    ax.plot(rs[start:].index, cum_balance, label='Bank (3% p.a.)', alpha=0.9, lw=LW, zorder=-1)

    # Aesthetics
    plt.axhline(1, linestyle='--', lw=LW, c='black', alpha=0.5, zorder=-1)
    sns.despine()
    ax.set_title('Portfolio Performance', fontsize=LARGE)
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%y')) 

    # Change the tick interval
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=len(comp.returns.index) // 150)) 
    
    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()
    
def plot_shares(comp, filename='single_stocks', save=False, scale=1):
    ''' Plots each individual share's value for each share in composition '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

    start = comp.portfolio_value.index[-1]
    for share, color in zip(comp.curr_shares, COLORS):
        tmp = share.timeline
        held = tmp[tmp.units > 0]
        start = min(start, held.index[0])

        ax.plot(held.index, (held.price / held.transaction_price.iloc[0]).to_list(), 
                label=share.code, alpha=0.9, lw=LW, c=color)

    # Aesthetics
    plt.axhline(1, xmax=0.865, linestyle='--', lw=LW, c='black', alpha=0.6, zorder=-1)
    sns.despine()
    ax.set_title('Single Stock Performance', fontsize=LARGE)
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL, loc='right')

    # Format the date into months & days
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%y')) 

    # Change the tick interval
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=len(comp.returns.index) // 150)) 

    # Puts x-axis labels on an angle
    #plt.gca().xaxis.set_tick_params(rotation = 30)  

    # Changes x-axis range
    held = comp.portfolio_value[comp.portfolio_value.index >= start].index
    diff = (held[-1] - held[0])
    ax.set_xbound(held[0] - pd.Timedelta(10*diff/300, 'days'), 
                  held[-1] + pd.Timedelta(5*diff/30, 'days'))

    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()