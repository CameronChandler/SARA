import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from matplotlib.ticker import FuncFormatter
from yahooquery import Ticker
from tqdm import tqdm
import csv

SMALL, MED, LARGE, LW = 18, 24, 30, 2
plt.rc('axes', titlesize=MED)    # fontsize of the axes title
plt.rc('axes', labelsize=MED)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL)   # fontsize of the tick labels
plt.rc('legend', fontsize=MED)   # legend fontsize
plt.rc('font', size=LARGE)         # controls default text sizes

TRADING_DAYS = 252
INITIAL_CASH = 20_000
TOMORROW = dt.datetime.today().date() + dt.timedelta(days=1)

TEST, PROD = 0, 1
SIIF = 1

class CashFlow:
    ''' Holds the meta data of one share in a composition '''
    
    def __init__(self, row):
        self.cash_flow = float(row['price'])
        self.date = pd.to_datetime(row['date'], format='%d/%m/%Y').date()
        
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return f'{self.date}: ${self.cash_flow}'
    
class Share:
    ''' Holds the meta data of one share in a composition '''
    
    def __init__(self, row, dates):
        self.code = row['code']
        self.timeline = pd.DataFrame({'units': 0, 'transaction_price': 0, 'fees': 0}, index=dates)
        
        # Merge in price data
        price = Ticker(f'{self.code}.AX').history(start=self.timeline.index[0], end=TOMORROW).reset_index()[['date', 'close']]
        price.columns = ['date', 'price']
        self.timeline = pd.merge(self.timeline, price, on='date', how='left').set_index('date')
        
    def add_info(self, row):
        date = pd.to_datetime(row['date'], format='%d/%m/%Y').date()
        self.timeline.loc[date:, 'units'] += int(row['units'])
        self.timeline.loc[date, 'fees'] = float(row['fee'])
        self.timeline.loc[date, 'transaction_price'] = float(row['price'])
        
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return f'{self.code}'

class Comp:
    ''' Represents a portfolio composition 
    name:            name of composition (shows up on graphs)
    filename:        filename for CSV file containing composition meta data
    shares:          list of Share objects
    daily:           pandas dataframe of daily close price for each share in composition
    portfolio_value: numpy array of net portfolio value over time'''
    
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.shares, self.cash_flows = self.load_comp()
        self.curr_shares = [share for share in self.shares if share.timeline.iloc[-1]['units'] != 0]
        self.portfolio_value = self.get_portfolio_value()
        self.returns = self.get_returns()
        
    def load_comp(self):
        ''' Loads units, price, sell_price, buy_date, sell_date and code as registered in the data csv '''
        # First, work out when the portfolio starts
        min_date = TOMORROW
        with open('./comps/'+self.filename+'.csv', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row['code']:
                    break
                min_date = min(min_date, pd.to_datetime(row['date'], format='%d/%m/%Y').date())
                
        dates = Ticker('NDQ.AX').history(start=min_date, end=TOMORROW).reset_index()['date']
        
        # Then parse the data
        shares = {}
        cash_flows = []
        with open('./comps/'+self.filename+'.csv', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row['code']
                if not code:
                    break
                elif code == 'CASH_FLOW':
                    cash_flows.append(CashFlow(row))
                else:
                    if code not in shares:
                        shares[code] = Share(row, dates)
                    shares[code].add_info(row)
                
        return list(shares.values()), cash_flows
    
    def get_portfolio_value(self, end_of_day=False):
        ''' Calculate the portfolio value for each day '''
        total_share_value = pd.Series(0, index=self.shares[0].timeline.index)
        for share in self.shares:
            total_share_value += share.timeline['price'] * share.timeline['units']
            
        return total_share_value + self.get_cash(end_of_day, total_share_value.index)
    
    def get_cash(self, end_of_day, dates):        
        ''' Return a numpy array representing the amount of cash in the portfolio at each day '''
        # Initialise starting balance
        # cash = [20_000, 20_000, 20_000, 20_000] 1xD 
        cash = pd.Series(self.cash_flows[0].cash_flow, index=dates)

        for cash_flow in self.cash_flows[1:]:
            # When should cash_flow count
            if end_of_day:
                effective_period = np.where(dates >  cash_flow.date , 1, 0)
            else:
                effective_period = np.where(dates >= cash_flow.date , 1, 0)

            cash += cash_flow.cash_flow * effective_period

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
            data.loc[cash_flow.date, 'cash_flow'] = cash_flow.cash_flow

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

def equidate_ax(fig, ax, dates, fmt="%Y-%m-%d", label="Date"):
    """
    Sets all relevant parameters for an equidistant date-x-axis.
    Tick Locators are not affected (set automatically)

    Args:
        fig: pyplot.figure instance
        ax: pyplot.axis instance (target axis)
        dates: iterable of datetime.date or datetime.datetime instances
        fmt: Display format of dates
        label: x-axis label
    Returns:
        None

    """        
    format_date = lambda index, pos: dates[np.clip(int(index + 0.5), 0, len(dates) - 1)].strftime(fmt)
    ax.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax.set_xlabel(label)
    fig.autofmt_xdate()

def plot_comps(comps, filename='strategy_comparison', save=False, scale=1):
    ''' Plots the performance of each comp '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    
    rs = comps[0].returns
    start = rs[rs.diff() != 0].index[1]
    
    ind = rs[start:].reset_index().index
    x = np.arange(min(ind)-1, max(ind)+1)
    for comp in comps:
        ax.plot(x, [1] + comp.returns[start:].to_list(), label=comp.name, alpha=0.9, lw=1.5*LW)
        
    # Plot Bank
    rate = 0.03
    cum_balance = [1] + list(np.cumprod((1+rate)**(1/TRADING_DAYS) * np.ones(len(x)-1)))
    ax.plot(x, cum_balance, label='Bank (3% p.a.)', alpha=0.9, lw=1.5*LW)
    
    # Aesthetics
    x = np.arange(-1, len(ind))
    ax.plot(x, np.ones(len(x)), linestyle='--', lw=LW, c='black', alpha=0.7)
    sns.despine()
    ax.set_title('Strategy Performance', fontsize=LARGE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL)
    equidate_ax(fig, ax, comps[0].returns[start:].index)
    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()
    
def plot_shares(comp, filename='single_stocks', save=False, scale=1):
    ''' Plots each individual share's value for each share in composition '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    
    start = comp.portfolio_value.index[0]
    
    for share in comp.curr_shares:
        tmp = share.timeline.reset_index(drop=True)
        held = tmp[tmp.units > 0]
        x = np.arange(min(held.index)-1, max(held.index)+1)
        ax.plot(x, [1] + (held.price / held.transaction_price.iloc[0]).to_list(), label=share.code, alpha=0.9, lw=LW)

    # Aesthetics
    x = np.arange(-1, len(comp.portfolio_value))
    ax.plot(x, np.ones(len(x)), linestyle='--', lw=LW, c='black', alpha=0.7)
    sns.despine()
    ax.set_title('Single Stock Performance', fontsize=LARGE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Value')
    ax.set_xlim(min(x) - 0.03*len(x), max(x) + 0.15*len(x))
    plt.legend(frameon=False, fontsize=SMALL, loc='right')
    equidate_ax(fig, ax, comp.portfolio_value.index)
    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()
    
def plot_profit(daily, PROFIT):
    ''' Deprecated '''
    raise(ImplementationError)
    
    fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)

    ax.plot(np.arange(len(daily)), PROFIT, lw=LW, c='green')
    ax.axhline(0, linestyle='--', lw=LW, c='black')

    ax.set_title('Profit', fontsize=LARGE)
    ax.set_xlabel('Date', fontsize=MED)
    ax.set_ylabel('Profit ($)', fontsize=MED)
    equidate_ax(fig, ax, daily.index)
    sns.despine()
    plt.show()
    
def plot_returns(daily, PROFIT):
    ''' Deprecated '''
    raise(ImplementationError)
    
    fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)

    profits = np.diff(PROFIT)
    x = np.arange(len(daily.index)-1)
    # Positive and Negative
    ax.bar(x, [max(0, p) for p in profits], width=1, color='green')
    ax.bar(x, [min(0, p) for p in profits], width=1, color='red')

    sns.despine()
    ax.set_title('Daily Returns ($)', fontsize=LARGE)
    ax.set_xlabel('Date', fontsize=MED)
    ax.set_ylabel('Daily Return ($)', fontsize=MED)
    equidate_ax(fig, ax, daily.index)
    plt.show()
    
################################ Modern Portfolio Theory ################################
    
def mean_variance_optimisation(returns, b_val=0.1, risk_free=False, interest_rate=0):
    ''' 
    Completes Mean Variance Optimisation
    
    INPUT:
        returns       - Dataframe where each column is an asset and each row is a date
        b_val         - Scalar, minimum expected rate of return
        risk_free     - Boolean, whether or not a risk free asset should be considered
        interest_rate - Float, annual rate of return of risk free asset
    OUTPUT:
        problem.value - Scalar, representing variance (i.e. squared volatility) of optimal asset allocation
        w.value       - Vector, representing optimal asset allocation 
    '''
    import cvxpy as cp

    μ = returns.mean().to_numpy() * TRADING_DAYS
    Σ = returns.cov().to_numpy() * TRADING_DAYS
    if risk_free:
        μ = np.append(μ, interest_rate)
        # Risk free assets have 0 variance, and 0 correlation with other instruments
        Σ = np.r_[Σ, [np.zeros(len(Σ))]] 
        Σ = np.c_[Σ, np.zeros(len(Σ))]  
    
    J = np.ones(len(μ))
    w = cp.Variable(len(μ))
    b = cp.Parameter(nonneg=True)
    # Minimum expected return
    b.value = b_val
    constraints = [μ.T @ w >= b,
                   J.T @ w == 1,
                   w >= 0]
    objective = cp.Minimize(cp.quad_form(w, Σ))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return problem.value, w.value.round(10)

################################ Sending Emails ################################
from email.mime.image import MIMEImage
from email.header import Header
import os
import smtplib
from email.mime.base import MIMEBase
from email import encoders
import cgi
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

def attach_image(img_dict):
    with open(img_dict['path'], 'rb') as file:
        msg_image = MIMEImage(file.read(), name=os.path.basename(img_dict['path']))
    msg_image.add_header('Content-ID', '<{}>'.format(img_dict['cid']))
    return msg_image

def attach_file(filename):
    part = MIMEBase('application', 'octect-stream')
    part.set_payload(open(filename, 'rb').read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename=%s' % os.path.basename(filename))
    return part

def send_email(msg, gmail_user, gmail_pwd, to):
    ''' Establishes connection and sends email `msg` '''
    mailServer = smtplib.SMTP('smtp.gmail.com', 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login(gmail_user, gmail_pwd)
    mailServer.sendmail(gmail_user, to, msg.as_string())
    mailServer.quit()
    
################################ Logging ################################  
def log(typ, running_level, name='', verbose=True, freq='WEEKLY'):
    ''' Log the emailing process. Log to log.txt if running_level is PROD, print if verbose is True '''
    # What is the message
    if typ == 'begin':
        msg = freq + '\n=== Begin running at ' + str(dt.datetime.now()) + ' ====\n'
    elif typ == 'success':
        msg = f'Email to {name.rjust(10)} succeeded at ' + str(dt.datetime.now()) + '\n'
    elif typ == 'failure':
        msg = f'Email to {name.rjust(10)} failed at ' + str(dt.datetime.now()) + '\n'
    elif typ == 'end':
        msg = '==== Completed emailing at ' + str(dt.datetime.now()) + ' ====\n\n'
    
    # Where to log
    if running_level == PROD:
        with open('log.txt', 'a') as f:
            f.write(msg)
    if verbose:
        print(msg)