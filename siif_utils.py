import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from yahooquery import Ticker
from tqdm import tqdm
import csv

SMALL, MED, LARGE, LW = 18, 24, 30, 3
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
        self.cash = float(row['price'])
        self.date = pd.to_datetime(row['date'], format='%d/%m/%Y').date()
        
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return f'{self.date}: ${self.cash}'
    
class Dividend(CashFlow):
    ''' Holds the meta data of one dividend payment '''
    
    def __init__(self, row):
        super().__init__(row)
    
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
    curr_shares:     list of Share objects that are currently owned
    cash_flows:      list of CashFlow objects
    dividends:       list of Dividend objects
    portfolio_value: numpy array of net portfolio value over time
    returns:         portfolio performance'''
    
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.cash_flows = []
        self.dividends  = []
        self.shares = self.load_comp()
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
        with open('./comps/'+self.filename+'.csv', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row['code']
                if not code:
                    break
                elif code == 'CASH_FLOW':
                    self.cash_flows.append(CashFlow(row))
                elif code == 'DIVIDEND':
                    self.dividends.append(Dividend(row))
                else:
                    if code not in shares:
                        shares[code] = Share(row, dates)
                    shares[code].add_info(row)
                
        return list(shares.values())
    
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
        cash = pd.Series(self.cash_flows[0].cash, index=dates)

        for cash_event in self.cash_flows[1:] + self.dividends:
            # When should cash_flow count
            if end_of_day:
                effective_period = np.where(dates >  cash_event.date , 1, 0)
            else:
                effective_period = np.where(dates >= cash_event.date , 1, 0)

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
        ax.plot(rs[start:].index, comp.returns[start:].to_list(), label=comp.name, alpha=0.9, lw=LW)

    # Plot Bank
    rate = 0.03
    cum_balance = list(np.cumprod((1+rate)**(1/TRADING_DAYS) * np.ones(len(rs[start:]))))
    ax.plot(rs[start:].index, cum_balance, label='Bank (3% p.a.)', alpha=0.9, lw=LW, zorder=-1)

    # Aesthetics
    plt.axhline(1, linestyle='--', lw=LW, c='black', alpha=0.5, zorder=-1)
    sns.despine()
    ax.set_title('Strategy Performance', fontsize=LARGE)
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%y')) 

    # Change the tick interval
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
    
    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()
    
def plot_shares(comp, filename='single_stocks', save=False, scale=1):
    ''' Plots each individual share's value for each share in composition '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

    start = comp.portfolio_value.index[-1]
    for share in comp.curr_shares:
        tmp = share.timeline
        held = tmp[tmp.units > 0]
        start = min(start, held.index[0])

        ax.plot(held.index, (held.price / held.transaction_price.iloc[0]).to_list(), label=share.code, alpha=0.9, lw=LW)

    # Aesthetics
    plt.axhline(1, xmax=0.865, linestyle='--', lw=LW, c='black', alpha=0.6, zorder=-1)
    sns.despine()
    ax.set_title('Single Stock Performance', fontsize=LARGE)
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL, loc='right')

    # Format the date into months & days
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%y')) 

    # Change the tick interval
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 

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