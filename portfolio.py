import numpy as np
import pandas as pd
import datetime as dt
from yahooquery import Ticker
from enum import Enum, auto

class PortfolioChoice(Enum):
    SIIF = auto()
    TEST = auto()

class CashFlow:
    ''' Holds the meta data of one cash_flow '''
    
    def __init__(self, date: dt.datetime, cash: float) -> None:
        self.date = date
        self.cash = cash
        
    def __str__(self) -> str:
        return self.__repr__()
        
    def __repr__(self) -> str:
        return f'<Cashflow {self.date}: ${self.cash}>'
    
class Dividend(CashFlow):
    ''' Holds the meta data of one dividend payment '''
    
    def __init__(self, date: dt.datetime, cash: float):
        super().__init__(date, cash)
        
    def __repr__(self) -> str:
        return f'<Dividend {self.date}: ${self.cash}>'
    
class Share:
    ''' Holds the meta data of one share in a portfolio 
        timeline: dataframe of share information
    '''
    
    def __init__(self, transactions: pd.DataFrame, dates: "pd.Series[dt.date]") -> None:
        self.code: str = transactions.iloc[0].code
        self.transactions = transactions.sort_values('date')
        self.dates = dates
        self.timeline = self._gen_timeline()
            
    def _gen_timeline(self) -> pd.DataFrame:
        # Initialise empty timeline
        timeline = pd.DataFrame({'units': 0, 'transaction_price': 0, 'fees': 0}, index=self.dates)
        
        timeline = self._add_transactions_to_timeline(timeline)
        
        timeline = self._add_prices_to_timeline(timeline)
        
        return timeline.fillna(0)
            
    def _add_transactions_to_timeline(self, timeline) -> pd.DataFrame:
        for _, row in self.transactions.iterrows():
            timeline.loc[row.date:, 'units'] += row.units
            timeline.loc[row.date, 'fees'] = row.fee
            timeline.loc[row.date, 'transaction_price'] = row.price
            
        return timeline
    
    def _add_prices_to_timeline(self, timeline, test=False) -> pd.DataFrame:
        
        prices = self._gen_real_prices()
        
        if prices is None:
            prices = self._gen_fake_prices()
        
        if test:
            return(timeline, prices)
        timeline = pd.merge(timeline, prices, on='date', how='left').ffill()
            
        return timeline
            
    def _gen_real_prices(self) -> pd.DataFrame:
        ''' Attempt to generate real prices. Returns None if prices cannot be found '''
        prices = Ticker(f'{self.code}.AX').history(start=self.dates[0])
        if isinstance(prices, dict):
            return None
        
        prices = prices.reset_index()[['date', 'close']].set_index('date')
        prices.columns = ['price']
        return prices
    
    def _gen_fake_prices(self) -> pd.DataFrame:
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

    def __str__(self) -> str:
        return self.__repr__()
        
    def __repr__(self) -> str:
        return f'{self.code}'

class Portfolio:
    ''' Represents a portfolio composition 
        name:            name of portfolio (shows up on graphs)
        filename:        filename for CSV file containing portfolio meta data
        shares:          list of Share objects
        curr_shares:     list of Share objects that are currently owned
        cash_flows:      list of CashFlow objects
        dividends:       list of Dividend objects
        portfolio_value: numpy array of net portfolio value over time
        returns:         portfolio performance
    '''
    
    def __init__(self, name: str, filename: str) -> None:
        self.name = name
        self.filename = filename
        self.shares, self.cash_flows, self.dividends = self.load_portfolio()
        self.curr_shares = [share for share in self.shares if share.timeline.iloc[-1]['units'] != 0]
        self.portfolio_value = self.get_portfolio_value()
        self.returns = self.get_returns()
        
    def load_portfolio(self) -> tuple[list[Share], list[CashFlow], list[Dividend]]:
        ''' Loads units, price, sell_price, buy_date, sell_date and code as registered in the data csv '''
        # First, work out when the portfolio starts
        data = pd.read_csv('./portfolios/'+self.filename+'.csv').dropna(how='all')
        data['date'] = pd.to_datetime(data.date, dayfirst=True).dt.date
        data['price'] = data.price.astype(float)
        min_date = data.date.min()
                
        self.dates = Ticker('NDQ.AX').history(start=min_date).reset_index()['date']

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
            cash_flows.append(CashFlow(row['date'], row['price']))
        
        for _, row in dividends_data.iterrows():
            dividends.append(Dividend(row['date'], row['price']))
                
        return shares, cash_flows, dividends
    
    def get_portfolio_value(self, end_of_day: bool=False) -> "pd.Series[float]":
        ''' Calculate the portfolio value for each day '''
        total_share_value = pd.Series(0, index=self.shares[0].timeline.index)
        for share in self.shares:
            total_share_value += share.timeline['price'] * share.timeline['units']
            
        return total_share_value + self.get_cash(end_of_day)
    
    def get_cash(self, end_of_day: bool) -> "pd.Series[float]":        
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
    
    def get_returns(self) -> "pd.Series[float]":
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

    def __str__(self) -> str:
        return self.__repr__()
        
    def __repr__(self) -> str:
        return f'{self.name}: {self.curr_shares}'

