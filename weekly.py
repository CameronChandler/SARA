############################################### TODO ################################################
# - Add to README

import pandas as pd
from emails import WeeklyEmail, Recipient, Image, send_email
from siif_utils import Comp, plot_shares, plot_comps, log, TEST, SIIF, PROD
import warnings
warnings.filterwarnings("ignore")
from time import sleep
import sys

############## INPUT PARAMETERS ############## 
args = sys.argv[1:]

if args and args[0] == 'prod':
    RUNNING_LEVEL = PROD
    PORTFOLIO = SIIF
else:
    RUNNING_LEVEL = TEST
    
    if args and args[1] == 'siif':
        PORTFOLIO = SIIF
    else:
        PORTFOLIO = TEST

#RUNNING_LEVEL = TEST
#PORTFOLIO = SIIF     
##############################################

log('begin', running_level=RUNNING_LEVEL)
####################################### STEP 1. PREPARE DATA ########################################

# Composition Name: Composition CSV file
test_comps = {'SIIF': 'TEST', 'NDQ': 'TEST_NDQ', 'A200': 'TEST_A200'}
siif_comps = {'SIIF': 'SIIF', 'NDQ': 'NDQ', 'A200': 'A200'} # 'MPT': 'MPT',  'SIIF Balanced': 'SIIF_MPT',
names = test_comps if PORTFOLIO == TEST else siif_comps
# ENSURE THAT THE MAIN PORTFOLIO IS ADDED TO `comps` FIRST
comps = [Comp(name, names[name]) for name in names]

######################################### STEP 2. PLOT DATA #########################################
plot_shares(comps[0], save=True)
plot_comps(comps, save=True)

######################################## STEP 3. SEND EMAILS ########################################
# Stock Alert Report Automaton
# SIIF Automated Reporting Assistant
# - Sara

TEST_NAME = 'Cameron'
        
# Load analyst emails
emails = pd.read_csv('./data/emails.csv')
recipients_test = emails[emails['name'] == TEST_NAME]
recipients_prod = emails
recipients = recipients_prod if RUNNING_LEVEL == PROD else recipients_test
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
        email = WeeklyEmail(email_address, recipient, images, comps[0].portfolio_value)
        send_email(email, email_address, email_password)
        log('success', name=recipient.name, running_level=RUNNING_LEVEL)
    except:
        log('failure', name=recipient.name, running_level=RUNNING_LEVEL)
    sleep(5)
        
log('end', running_level=RUNNING_LEVEL)