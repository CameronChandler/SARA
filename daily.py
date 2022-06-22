import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates
import pandas as pd
from yahooquery import Ticker
from emails import DailyEmail, Recipient, Image, send_email
from siif_utils import log, TEST, PROD, TOMORROW, COLORS
import warnings
warnings.filterwarnings("ignore")
from time import sleep
import numpy as np
import sys

############## INPUT PARAMETERS ############## 
args = sys.argv[1:]

if args and args[0] == 'prod':
    RUNNING_LEVEL = PROD
else:
    RUNNING_LEVEL = TEST

RUNNING_LEVEL = TEST
##############################################

####################################### STEP 1. PREPARE DATA ########################################
log('begin', running_level=RUNNING_LEVEL, freq='DAILY')

TODAY = dt.date.today()
LAST_WEEK = TODAY - pd.Timedelta(days=7)

def load_data(codes, start=str(LAST_WEEK), end=TOMORROW):
    ''' Takes list of shares and returns data from the start date '''
    daily = pd.DataFrame({'date': []})

    # Load data
    for code in codes:
        df = Ticker(f'{code}.AX').history(start=start, end=end).reset_index()[['date', 'close']]
        df.columns = ['date', code]
        daily = pd.merge(daily, df, on='date', how='outer')

    daily['date'] = pd.to_datetime(daily.date)
    daily = daily.sort_values('date').ffill().set_index('date')
    return daily

######################################### STEP 2. PLOT DATA #########################################
SMALL, MED, LARGE, LW = 18, 24, 30, 2
plt.rc('axes', titlesize=MED)    # fontsize of the axes title
plt.rc('axes', labelsize=MED)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL)   # fontsize of the tick labels
plt.rc('legend', fontsize=MED)   # legend fontsize
plt.rc('font', size=LARGE)         # controls default text sizes

def plot_shares(daily: pd.DataFrame, filename: str, scale: float=1.0):
    ''' Plots each individual share's value for each share in composition '''
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
    
######################################## STEP 3. SEND EMAILS ########################################
TEST_NAME = 'Cameron'

# Load analyst emails
emails = pd.read_csv('./data/emails.csv')
recipients_test = emails[emails['name'] == TEST_NAME]
recipients_prod = emails
recipients_data = recipients_prod if RUNNING_LEVEL == PROD else recipients_test
recipients: list[Recipient] = []
for _, row in recipients_data.iterrows():
    codes: list[str] = [] if row['stocks'] is np.nan else row['stocks'].split()
    recipients.append(Recipient(row['email'], row['name'], codes, float(row['sensitivity'])))

email_address = "sarasiifbot@gmail.com"
with open('./data/password.txt') as fp:
    email_password = fp.read()

LOGO_SCALE = (258, 155)
GRAPH_SCALE = (691, 389)

try:
    for recipient in recipients:

        if not recipient.codes:
            continue

        daily = load_data(recipient.codes + ['A200', 'NDQ'])

        # Ensure that the markets ran today
        if TODAY != daily.index[-1].date():
            break

        daily_changes = []
        for code in recipient.codes:
            pct_change = 100 * (daily[code].iloc[-1] / daily[code].iloc[-2] - 1)
            if abs(pct_change) > float(row['sensitivity']):
                daily_changes.append([code, pct_change])

        if not daily_changes:
            continue

        filename = 'daily'
        plot_shares(daily, filename)

        all_files = [(f'./images/{filename}.png', GRAPH_SCALE), 
                     ('./images/SIIF Logo.png', LOGO_SCALE)]

        images: list[Image] = [Image(path, width, height) for path, (width, height) in all_files]

        try:
            email = DailyEmail(email_address, recipient, images, daily_changes)
            send_email(email, email_address, email_password)
            log('success', name=recipient.name, running_level=RUNNING_LEVEL)
        except:
            log('failure', name=recipient.name, running_level=RUNNING_LEVEL)

        sleep(5)
except:
    log('error', name='Error', running_level=RUNNING_LEVEL)
finally:
    log('end', running_level=RUNNING_LEVEL)