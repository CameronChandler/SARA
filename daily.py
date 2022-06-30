import datetime as dt
import pandas as pd
from yahooquery import Ticker
from emails import DailyEmail, Recipient, Image, send_email
from logger import Logger, RunningLevel, Date
from plotting import plot_shares
import warnings
warnings.filterwarnings("ignore")
from time import sleep
import numpy as np
import sys

############## INPUT PARAMETERS ############## 
args = sys.argv[1:]

if args and args[0] == 'prod':
    running_level = RunningLevel.PROD
else:
    running_level = RunningLevel.TEST

running_level = RunningLevel.TEST

log = Logger('DAILY', running_level)
log.begin()
####################################### STEP 1. PREPARE DATA ########################################

TODAY = dt.date.today()
LAST_WEEK = TODAY - pd.Timedelta(days=7)

def load_data(codes: list[str], start: dt.date=Date.last_week(), end: dt.date=Date.tomorrow()) -> pd.DataFrame:
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

######################################## STEP 3. SEND EMAILS ########################################
TEST_NAME = 'Cameron'

# Load analyst emails
emails = pd.read_csv('./data/emails.csv')
recipients_test = emails[emails['name'] == TEST_NAME]
recipients_prod = emails
recipients_data = recipients_prod if running_level == RunningLevel.PROD else recipients_test
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
                daily_changes.append((code, pct_change))

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
            log.success(recipient.name)
        except:
            log.failure(recipient.name)

        sleep(5)
except:
    log.error(name='Error')
finally:
    log.end()