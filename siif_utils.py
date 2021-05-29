import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from matplotlib.ticker import FuncFormatter
from yahooquery import Ticker
from tqdm import tqdm
import cvxpy as cp
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


class Share:
    ''' Holds the meta data of one share in a composition '''
    
    def __init__(self, meta_data):
        # meta_data structure: [code, units, buy_price, sell_price, buy_date, sell_date, buy_fee, sell_fee]
        self.code = meta_data[0]
        self.units = int(meta_data[1])
        self.buy_price  = float(meta_data[2])
        self.sell_price = None if meta_data[3] == '-1' else float(meta_data[3])
        self.buy_date  = pd.to_datetime(meta_data[4], format='%d/%m/%Y').date()
        self.sell_date = TOMORROW if meta_data[5] == '-1' else pd.to_datetime(meta_data[5], format='%d/%m/%Y')
        self.buy_fee  = float(meta_data[6])
        self.sell_fee = float(meta_data[7])
        
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
        self.shares = self.load_comp()
        self.daily = self.load_data()
        self.portfolio_value = self.portfolio_value()
        
    def load_comp(self):
        ''' Loads units, price, sell_price, buy_date, sell_date and code as registered in the data csv '''
        shares = []
        with open('./comps/'+self.filename+'.csv', newline='') as f:
            next(csv.reader(f, delimiter=' ', quotechar='|'))
            reader = csv.reader(f, delimiter=' ', quotechar='|')
            for row in reader:
                shares.append(Share(row[0].split(','))) 
                
        return shares
                
    def load_data(self, start='2021-04-23', end='3000-01-01', verbose=True):
        ''' Takes list of shares and returns data from the start date '''
        if verbose:
            print('Loading', self.name)
            
        daily = pd.DataFrame({'date': []})
        codes = [share.code for share in self.shares]
        codes = tqdm(codes) if verbose else codes
        
        # Load data
        for code in codes:
            df = Ticker(f'{code}.AX').history(start=start, end=end).reset_index()[['date', 'close']]
            df.columns = ['date', code]
            daily = pd.merge(daily, df, on='date', how='outer')
            
        daily['date'] = pd.to_datetime(daily.date)
        daily = daily.sort_values('date').ffill().set_index('date')
        
        # Set 0s where shares were not owned
        for i, share in enumerate(self.shares):
            daily.iloc[:daily.index.to_list().index(share.buy_date), i] = 0
            if share.sell_date != TOMORROW:
                daily.iloc[daily.index.to_list().index(share.sell_date):, i] = 0

        return daily
    
    def portfolio_value(self):
        ''' Calculate the portfolio value for each day '''
        units = np.array([share.units for share in self.shares])
        return self.daily.to_numpy() @ units + self.get_cash()
    
    def get_cash(self):        
        ''' Return a numpy array representing the amount of cash in the portfolio at each day '''
        daily = self.daily.copy().reset_index()

        # cash = [20_000, 20_000, 20_000, 20_000] 1xD 
        cash = INITIAL_CASH*np.ones(len(daily))

        for share in self.shares:
            held_onwards = np.where(daily.date >= share.buy_date , 1, 0)
            sold_onwards = np.where(daily.date >= share.sell_date, 1, 0)
            # cash subtract cost from cash forever [20_000, 18_000, 18_000, 18_000] 1xD (-2000)
            cash -= held_onwards * (share.units*share.buy_price + share.buy_fee)

            # cash add money back from selling shares forever [20_000, 18_000, 21_000, 21_000] (+3000)
            if share.sell_date != TOMORROW:
                cash += sold_onwards * (share.units*share.sell_price - share.sell_fee)

        return cash
                
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return f'{self.name}: {self.shares}'

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

def plot_shares(comp, filename='single_stocks', save=False, scale=1):
    ''' Plots each individual share's value for each share in composition '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    
    for share in comp.shares:
        tmp = comp.daily.reset_index(drop=True)
        ind = tmp[tmp[share.code] > 0].index
        x = np.arange(min(ind)-1, max(ind)+1)
        ax.plot(x, [1] + list(tmp[tmp[share.code] > 0][share.code] / share.buy_price), label=f'{share.code}', alpha=0.9, lw=LW)

    x = np.arange(len(comp.daily))
    ax.plot(x, np.ones(len(comp.daily)), linestyle='--', lw=LW, c='black', alpha=0.7)
    sns.despine()
    ax.set_title('Single Stock Performance', fontsize=LARGE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Value')
    ax.set_xlim(-0.05*max(x), 1.15*max(x))
    plt.legend(frameon=False, fontsize=SMALL, loc='right')
    equidate_ax(fig, ax, comp.daily.index)
    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()
    
def plot_comps(comps, filename='strategy_comparison', save=False, scale=1):
    ''' Plots the performance of each comp '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    
    ind = comps[0].daily.reset_index(drop=True).index
    x = np.arange(min(ind)-1, max(ind)+1)
    for comp in comps:
        ax.plot(x, [1] + list(comp.portfolio_value / INITIAL_CASH), label=comp.name, alpha=0.9, lw=1.5*LW)

    x = np.arange(-1, len(ind))
    ax.plot(x, np.ones(len(x)), linestyle='--', lw=LW, c='black', alpha=0.7)
    sns.despine()
    ax.set_title('Strategy Performance', fontsize=LARGE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL)
    equidate_ax(fig, ax, comps[0].daily.index)
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