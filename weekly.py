# Stock Alert Report Automaton
# SIIF Automated Reporting Assistant
# - SARA

import pandas as pd
from emails import WeeklyEmail, Recipient, Image, send_email, GRAPH_SCALE
from portfolio import Portfolio, PortfolioChoice
from logger import Logger
from plotting import plot_portfolio, plot_portfolios
import warnings
warnings.filterwarnings("ignore")
from time import sleep
from typing import List

############## INPUT PARAMETERS ############## 
portfolio_choice = PortfolioChoice.TEST

log = Logger('WEEKLY')
log.begin()
####################################### STEP 1. PREPARE DATA ########################################

# Portfolio Name: Portfolio CSV file
names = {'portfolio': 'portfolio', 'NDQ': 'TEST_NDQ', 'A200': 'TEST_A200'}
# ENSURE THAT THE MAIN PORTFOLIO IS ADDED TO `portfolios` FIRST
portfolios = [Portfolio(name, names[name]) for name in names]

######################################### STEP 2. PLOT DATA #########################################
plot_portfolio(portfolios[0])
plot_portfolios(portfolios)

######################################## STEP 3. SEND EMAILS ########################################
        
# Load analyst emails
emails = pd.read_csv('./data/emails.csv') # type: ignore
recipients = [Recipient(row['email'], row['name']) for _, row in emails.iterrows()]

with open('./data/bot email.txt') as fp:
    email_address = fp.read()
    
with open('./data/password.txt') as fp:
    email_password = fp.read()

# Make sure the logo is last in the list
all_files = [('./images/strategy_comparison.png', GRAPH_SCALE), 
             ('./images/single_stocks.png', GRAPH_SCALE)]

images: List[Image] = [Image(path, width, height) for path, (width, height) in all_files]

for recipient in recipients:
    try:
        email = WeeklyEmail(email_address, recipient, images, portfolios[0].portfolio_value)
        send_email(email, email_address, email_password)
        log.success(recipient.name)
    except:
        log.failure(recipient.name)
    sleep(5)
        
log.end()