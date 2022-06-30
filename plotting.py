import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from portfolio import Portfolio

SMALL, MED, LARGE, LW = 18, 24, 30, 3
plt.rc('axes', titlesize=MED)    # fontsize of the axes title
plt.rc('axes', labelsize=MED)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL) # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL) # fontsize of the tick labels
plt.rc('legend', fontsize=MED)   # legend fontsize
plt.rc('font', size=LARGE)       # controls default text sizes

TRADING_DAYS = 252

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', '#ffe119', '#46f0f0', '#008080', 
          '#9a6324', '#808000', '#000075', '#aaffc3', '#000000', '#808080']

def plot_portfolios(portfolios: list[Portfolio], filename: str='strategy_comparison') -> None:
    ''' Plots the performance of each portfolio '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

    rs = portfolios[0].returns
    start = rs[rs.diff() != 0].index[1]

    for portfolio in portfolios:
        x = rs[start:]
        y = portfolio.returns[start:]
        ax.plot(x[~x.index.duplicated(keep='first')].index, y[~y.index.duplicated(keep='first')], label=portfolio.name, alpha=0.9, lw=LW)

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
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=len(portfolio.returns.index) // 150)) 
    
    plt.savefig('./images/'+filename+'.png', dpi=2*fig.dpi)
    
def plot_portfolio(portfolio: Portfolio, filename: str='single_stocks') -> None:
    ''' Plots each individual share's value for each share in portfolio '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

    start = portfolio.portfolio_value.index[-1]
    for share, color in zip(portfolio.curr_shares, COLORS):
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
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=len(portfolio.returns.index) // 150)) 

    # Puts x-axis labels on an angle
    #plt.gca().xaxis.set_tick_params(rotation = 30)  

    # Changes x-axis range
    held = portfolio.portfolio_value[portfolio.portfolio_value.index >= start].index
    diff = (held[-1] - held[0])
    ax.set_xbound(held[0] - pd.Timedelta(10*diff/300, 'days'), 
                  held[-1] + pd.Timedelta(5*diff/30, 'days'))

    plt.savefig('./images/'+filename+'.png', dpi=2*fig.dpi)

def plot_shares(daily: pd.DataFrame, filename: str, scale: float=1.0) -> None:
    ''' Plots each individual share's value for each share in dataframe '''
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    
    x = daily.index
    for code, color in zip(daily, COLORS):
        ls = '--' if code in ['A200', 'NDQ'] else '-'
        ax.plot(x, daily[code] / daily[code].iloc[0], label=code, alpha=0.9, lw=1.5*LW, linestyle=ls, c=color)

    # Aesthetics
    ax.plot(x, np.ones(len(daily)), linestyle='--', lw=LW, c='black', alpha=0.7)
    sns.despine()
    ax.set_title('Weekly Single Stock Performance', fontsize=LARGE)
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Value')
    #ax.set_xlim(min(x) - 0.03*len(x), max(x) + 0.15*len(x))
    ax.legend(frameon=False, fontsize=SMALL, loc='upper left')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m')) 
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1)) 
    plt.savefig('./images/'+filename+'.png', dpi=scale*2*fig.dpi)