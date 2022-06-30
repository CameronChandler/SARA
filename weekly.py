import pandas as pd
from emails import WeeklyEmail, Recipient, Image, send_email
from portfolio import Portfolio, PortfolioChoice
from logger import Logger, RunningLevel
from plotting import plot_portfolio, plot_portfolios
import warnings
warnings.filterwarnings("ignore")
from time import sleep
import sys

############## INPUT PARAMETERS ############## 
args = sys.argv[1:]

if args and args[0] == 'prod':
    running_level = RunningLevel.PROD
    portfolio_choice = PortfolioChoice.SIIF
else:
    running_level = RunningLevel.TEST
    
    if args and args[1] == 'siif':
        portfolio_choice = PortfolioChoice.SIIF
    else:
        portfolio_choice = PortfolioChoice.TEST

running_level = RunningLevel.TEST
portfolio_choice = PortfolioChoice.SIIF     

log = Logger('WEEKLY', running_level)
log.begin()
####################################### STEP 1. PREPARE DATA ########################################

# Portfolio Name: Portfolio CSV file
test_portfolios = {'SIIF': 'TEST', 'NDQ': 'TEST_NDQ', 'A200': 'TEST_A200'}
siif_portfolios = {'SIIF': 'SIIF', 'NDQ': 'NDQ', 'A200': 'A200'} # 'MPT': 'MPT',  'SIIF Balanced': 'SIIF_MPT',
names = test_portfolios if portfolio_choice == PortfolioChoice.TEST else siif_portfolios
# ENSURE THAT THE MAIN PORTFOLIO IS ADDED TO `portfolios` FIRST
portfolios = [Portfolio(name, names[name]) for name in names]

######################################### STEP 2. PLOT DATA #########################################
plot_portfolio(portfolios[0])
plot_portfolios(portfolios)

######################################## STEP 3. SEND EMAILS ########################################
# Stock Alert Report Automaton
# SIIF Automated Reporting Assistant
# - Sara

TEST_NAME = 'Cameron'
        
# Load analyst emails
emails = pd.read_csv('./data/emails.csv')
recipients_test = emails[emails['name'] == TEST_NAME]
recipients_prod = emails
recipients = recipients_prod if running_level == RunningLevel.PROD else recipients_test
recipients = [Recipient(row['email'], row['name']) for _, row in recipients.iterrows()]

email_address = "sarasiifbot@gmail.com"
with open('./data/password.txt') as fp:
    email_password = fp.read()

LOGO_SCALE = (258, 155)
GRAPH_SCALE = (691, 389)

# Make sure the logo is last in the list
all_files = [('./images/strategy_comparison.png', GRAPH_SCALE), 
             ('./images/single_stocks.png', GRAPH_SCALE),
             ('./images/SIIF Logo.png', LOGO_SCALE)]

images: list[Image] = [Image(path, width, height) for path, (width, height) in all_files]

for recipient in recipients:
    try:
        email = WeeklyEmail(email_address, recipient, images, portfolios[0].portfolio_value)
        send_email(email, email_address, email_password)
        log.success(recipient.name)
    except:
        log.failure(recipient.name)
    sleep(5)
        
log.end()