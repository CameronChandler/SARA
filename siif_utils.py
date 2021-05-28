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
UNITS, BUY_PRICE, BUY_DATE, SELL_DATE, SELL_PRICE, FEE = 0, 1, 2, 3, 4, 5
INITIAL_CASH = 20_000
TOMORROW = dt.datetime.today().date() + dt.timedelta(days=1)

CODE = 0
def load_comp(filename):
    ''' Loads units, price, sell_price, buy_date, sell_date and code as registered in the data csv '''
    comp = {}
    with open('./comps/'+filename+'.csv', newline='') as f:
        next(csv.reader(f, delimiter=' ', quotechar='|'))
        reader = csv.reader(f, delimiter=' ', quotechar='|')
        for row in reader:
            row = row[0].split(',')
            code = row[CODE]
            row = row[1:]
            
            # Convert units and prices to int/float
            row[UNITS] = int(row[UNITS])
            row[BUY_PRICE] = float(row[BUY_PRICE])
            row[SELL_PRICE] = float(row[SELL_PRICE])
            row[BUY_DATE] = row[BUY_DATE].split('/')[::-1]
            row[BUY_DATE][2] = f'{int(row[BUY_DATE][2]):02d}'
            row[BUY_DATE] = '-'.join(row[BUY_DATE])
            row[SELL_DATE] = TOMORROW if row[SELL_DATE] == '-1' else pd.to_datetime(row[SELL_DATE])
            row[FEE] = float(row[FEE])
            comp[code] = row
            
    return comp

def load_comps(filenames, names):
    ''' Takes csv comp filename and readable name and returns `comps` object '''
    return {name: {'comp': load_comp(filename)} for filename, name in zip(filenames, names)}

def get_cash(comp):        
    ''' Return a numpy array representing the amount of cash in the portfolio at each day '''
    daily = comp['daily'].copy().reset_index()
    
    # cash = [20_000, 20_000, 20_000, 20_000] 1xD 
    cash = INITIAL_CASH*np.ones(len(daily))
    
    for code, data in comp['comp'].items():
        # cash subtract cost from cash forever [20_000, 18_000, 18_000, 18_000] 1xD (-2000)
        cash -= data[UNITS]*data[BUY_PRICE]*np.where(daily.date.isin(daily[(daily.date >= data[BUY_DATE])].date), 1, 0)
        
        # cash add money back from selling shares forever [20_000, 18_000, 21_000, 21_000] (+3000)
        cash -= data[UNITS]*data[SELL_PRICE]*np.where(daily.date.isin(daily[(daily.date >= data[SELL_DATE])].date), 1, 0)
        
    return cash

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
    

def load_data(codes, start='2020-07-27', end='3000-01-01', verbose=True):
    ''' Takes list of shares and returns data from the start date '''
    daily = pd.DataFrame({'date': []})

    codes = tqdm(codes) if verbose else codes
    print('Loading Data') if verbose else None
    for code in codes:
        df = Ticker(f'{code}.AX').history(start=start, end=end).reset_index()

        df = df[['date', 'close']]
        df.columns = ['date', code]
        daily = pd.merge(daily, df, on='date', how='outer').sort_values('date')

    daily['date'] = pd.to_datetime(daily.date)
    daily = daily.ffill().set_index('date')
    return daily

def plot_shares(daily, SHARES, filename='single_stocks', save=False, scale=1):
    ''' Plots each individual share's value '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    
    for code in daily:
        tmp = daily.reset_index(drop=True)
        ind = tmp[tmp[code] > 0].index
        x = np.arange(min(ind)-1, max(ind)+1)
        ax.plot(x, [1] + list(tmp[tmp[code] > 0][code] / SHARES[code][BUY_PRICE]), label=f'{code}', alpha=0.9, lw=LW)

    x = np.arange(len(daily))
    ax.plot(x, np.ones(len(daily)), linestyle='--', lw=LW, c='black', alpha=0.7)
    sns.despine()
    ax.set_title('Single Stock Performance', fontsize=LARGE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL)
    equidate_ax(fig, ax, daily.index)
    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()
    
def plot_comps(comps, filename='strategy_comparison', save=False, scale=1):
    ''' Plots the performance of each comp '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    
    for comp_name, comp in comps.items():
        daily = comp['daily']
        ind = daily.reset_index(drop=True).index
        x = np.arange(min(ind)-1, max(ind)+1)
        ax.plot(x, [1] + list(comp['portfolio_value'] / INITIAL_CASH), label=comp_name, alpha=0.9, lw=1.5*LW)

    x = np.arange(-1, len(daily))
    ax.plot(x, np.ones(len(x)), linestyle='--', lw=LW, c='black', alpha=0.7)
    sns.despine()
    ax.set_title('Strategy Performance', fontsize=LARGE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Value')
    plt.legend(frameon=False, fontsize=SMALL)
    equidate_ax(fig, ax, daily.index)
    if save:
        plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)
    plt.show()
    
def plot_profit(daily, PROFIT):
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

#### SENDING EMAILS
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