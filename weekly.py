############################################### TODO ################################################
# - Add to README

import datetime as dt
import pandas as pd
from siif_utils import Comp, plot_shares, plot_comps, send_email, attach_file, attach_image, log, TEST, SIIF, PROD
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

import html
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

TEST_NAME = 'Cameron'
        
# Load analyst emails
emails = pd.read_csv('./data/emails.csv')
ANALYSTS_TEST = emails[emails['name'] == TEST_NAME]
ANALYSTS_PROD = emails
ANALYSTS = ANALYSTS_PROD if RUNNING_LEVEL == PROD else ANALYSTS_TEST

gmail_user = "sarasiifbot@gmail.com"
with open('./data/password.txt') as f:
    gmail_pwd = f.read()

LOGO_PATH = './images/SIIF Logo.png'
LOGO_SCALE = (258, 155)
GRAPH_SCALE = (691, 389)

def generate_email(gmail_user, to, img_dict, name='Analyst'):
    msg = MIMEMultipart('related')
    msg['Subject'] = Header(u'SIIF Weekly Report', 'utf-8')
    msg['From'] = gmail_user
    msg['To'] = to
    msg_alternative = MIMEMultipart('alternative')
    msg_text = MIMEText(u'Image not working', 'plain', 'utf-8')
    msg_alternative.attach(msg_text)
    msg.attach(msg_alternative)

    msg_html = u'<p>Dear {},</p>'.format(name)
    msg_html += u'<p>Here\'s an update on the SIIF portfolio:</p>'
    
    # Strategy Comparison
    portfolio = comps[0].portfolio_value
    last_week_date = portfolio.index[-1] - pd.Timedelta(days=7)
    last_week_value = portfolio[portfolio.index <= last_week_date][-1]
    pct = round(100*(portfolio[-1]/last_week_value - 1), 1)
    msg_html += u'<p>The current portfolio value is <b>${}</b>, that is <b>{}% {}</b> from last week.</p>'.format(
        round(portfolio[-1], 2), abs(pct), "up" if pct >= 0 else "down")
    
    msg_html += '<div dir="ltr">''<img src="cid:{cid}" alt="{alt}" width="{w}" height="{h}"><br></div>'.format(
                alt=html.escape('image not found', quote=True), w=GRAPH_SCALE[0], h=GRAPH_SCALE[1], **img_dict[0])
    msg_html += u"<p>The above graph compares SIIF's current portfolio against several other strategies. They are:</p>"
    msg_html += u"<p>Investing entirely in the NASDAQ 100, ASX 200, or into a 3% p.a. savings account.</p>"
    
    # SIIF Current Breakdown
    msg_html += u"<p>Here is the breakdown of SIIF's portfolio:</p>"
    
    msg_html += '<div dir="ltr">''<img src="cid:{cid}" alt="{alt}" width="{w}" height="{h}"><br></div>'.format(
                alt=html.escape('image not found', quote=True), w=GRAPH_SCALE[0], h=GRAPH_SCALE[1], **img_dict[1])
    
    # Add sign-off and logo
    msg_html += '<p>Have a great day!</p><p>From Sara (SIIF Automated Reporting Assistant)</p>'
    msg_html += '<div dir="ltr">''<img src="cid:{cid}" alt="{alt}" width="{w}" height="{h}"><br></div>'.format(
                alt=html.escape('image not found', quote=True), w=LOGO_SCALE[0], h=LOGO_SCALE[1], **img_dict[-1])
    
    msg_html += u"<p>---------------------------------</p>"
    msg_html += u"<p><small>Do not reply to this email</small></p>"
    msg_html += u"<p><small>Code available at https://github.com/CameronChandler/SARA</small></p>"
    msg_html += u"<p><small>Disclaimer: This email is automated and the data/visualisations/calculations are subject to errors!</small></p>"
    msg_html += u"<p><small>This has not been checked by a human, so please do not use to inform your financial decisions.</small></p>"

    msg_html = MIMEText(msg_html, 'html', 'utf-8')
    msg_alternative.attach(msg_html)
    for img in img_dict:
        msg.attach(attach_image(img))

    return msg

img_dict = []
# Make sure the logo is last in the list
all_files = [('./images/strategy_comparison.png', GRAPH_SCALE), ('./images/single_stocks.png', GRAPH_SCALE),
             (LOGO_PATH, LOGO_SCALE)]

for file, scale in all_files:
    img_dict_single = dict(title=file, path=file, cid=str(uuid.uuid4()), scale=scale)
    img_dict.append(img_dict_single)

for i, row in ANALYSTS.iterrows():
    name, email = row['name'], row['email']
    if RUNNING_LEVEL == TEST and name != TEST_NAME:
        continue

    try:
        email_msg = generate_email(gmail_user, email, img_dict=img_dict, name=name)
        send_email(email_msg, gmail_user, gmail_pwd, email)
        log('success', name=name, running_level=RUNNING_LEVEL)
    except:
        log('failure', name=name, running_level=RUNNING_LEVEL)
    sleep(5)
        
log('end', running_level=RUNNING_LEVEL)