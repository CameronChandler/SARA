import os
import html
import uuid
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import datetime
from typing import List, Tuple

from email.header import Header
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

@dataclass
class Recipient:
    ''' Class representing a SIIF recipient of SARA emails '''
    email_address: str
    name: str = 'SIIF Member'
    codes: List[str] = field(default_factory=list)
    sensitivity: float = float('inf')

LOGO_SCALE = (258, 155)
GRAPH_SCALE = (691, 389)

@dataclass
class Image:
    ''' Class representing an image to be included in an email '''
    path: str
    width: int
    height: int
    cid: str = field(init=False)
    graphic: MIMEImage = field(init=False)
    
    def __post_init__(self) -> None:
        self.cid = str(uuid.uuid4())

        with open(self.path, 'rb') as fp:
            graphic = MIMEImage(fp.read(), name=os.path.basename(self.path))
        graphic.add_header('Content-ID', f'<{self.cid}>')
        self.graphic = graphic


class Email(ABC):
    ''' Abstract class for email. Responsible for creating an email '''
    header: str

    def __init__(self, email_address: str, recipient: Recipient, images: List[Image]) -> None:
        self.email_address = email_address
        self.recipient = recipient
        # We assume that logo is the last image in the images list
        self.images = images[:-1]
        self.logo = images[-1]
        self.email = self._create_email()

    @abstractmethod
    def _create_html_body(self, recipient: Recipient) -> MIMEText:
        ...
        
    def _create_image_html(self, image: Image) -> str:
        return f'''<div dir="ltr"><img src="cid:{image.cid}" 
        alt="{html.escape('image not found', quote=True)}" 
        width="{image.width}" height="{image.height}"><br></div>'''

    def _generate_alternative(self) -> MIMEMultipart:
        alternative = MIMEMultipart('alternative')
        text = MIMEText(u'Image not working', 'plain', 'utf-8')
        alternative.attach(text)
        return alternative

    def _generate_message(self, recipient: Recipient, header: str) -> MIMEMultipart:
        msg = MIMEMultipart('related')
        msg['Subject'] = Header(header, 'utf-8')
        msg['From'] = self.email_address
        msg['To'] = recipient.email_address
        return msg

    def _create_signoff_html(self) -> str:
        return f'''<p>Have a great day!</p><p>From Sara (SIIF Automated Reporting Assistant)</p>
        {self._create_image_html(self.logo)}
        <p>---------------------------------</p>
        <p><small>Do not reply to this email</small></p>
        <p><small>Code available at https://github.com/CameronChandler/SARA</small></p>
        <p><small>Disclaimer: This email is automated and the data/visualisations/calculations are subject to errors!</small></p>
        <p><small>This has not been checked by a human, so please do not use to inform your financial decisions.</small></p>'''
    
    def _create_email(self) -> str:
        msg = self._generate_message(self.recipient, self.header)
        alt = self._generate_alternative()
        msg.attach(alt)

        alt.attach(self._create_html_body(self.recipient))
        for image in self.images + [self.logo]:
            msg.attach(image.graphic)

        return msg.as_string()


class WeeklyEmail(Email):
    ''' Create and send weekly emails '''
    header = 'SIIF Weekly Report'

    def __init__(
        self, email_address: str, recipient: Recipient, images: List[Image], portfolio: "pd.Series[float]"
    ) -> None:
        self.portfolio = portfolio
        super().__init__(email_address, recipient, images)
    
    def _create_html_body(self, recipient: Recipient) -> MIMEText:

        # Strategy Comparison
        last_week_date: datetime.date = self.portfolio.index[-1] - pd.Timedelta(days=7)
        last_week_portfolio_value: np.float64 = self.portfolio[self.portfolio.index <= last_week_date][-1]
        pct_change: float = round(100*(self.portfolio[-1]/last_week_portfolio_value - 1), 1)

        msg_html = f'''<p>Dear {recipient.name},</p>
        <p>Here\'s an update on the SIIF portfolio:</p>
        <p>The current portfolio value is <b>${round(self.portfolio[-1], 2)}</b>, 
        that is <b>{abs(pct_change)}% {"up" if pct_change >= 0 else "down"}</b> from last week.</p>
        {self._create_image_html(self.images[0])}
        <p>The above graph compares SIIF's current portfolio against several other strategies. They are:</p>
        <p>Investing entirely in the NASDAQ 100, ASX 200, or into a 3% p.a. savings account.</p>
        <p>Here is the breakdown of SIIF's portfolio:</p>
        {self._create_image_html(self.images[1])}
        {self._create_signoff_html()}'''

        return MIMEText(msg_html, 'html', 'utf-8')


class DailyEmail(Email):
    ''' Create and send daily emails '''
    header = 'SIIF - Large Stock Movement Alert'

    def __init__(self, email_address: str, recipient: Recipient, images: List[Image], daily_changes: List[Tuple[str, float]]) -> None:
        self.daily_changes = daily_changes
        super().__init__(email_address, recipient, images)

    def _daily_changes_html(self) -> str:
        out = ''
        for code, change in self.daily_changes:
            out += f"<p>Today {code} <b>{'dropped' if change < 0 else 'rose'} {round(abs(change), 1)}</b>%</p>"
        return out

    def _create_html_body(self, recipient: Recipient) -> MIMEText:

        first_code = self.daily_changes[0][0]
        msg_html = f'''<p>Dear {recipient.name},</p>
        {self._daily_changes_html()}
        <p>You may like to investigate why. <a href="https://www.google.com/search?q={first_code}+asx">Click here to begin researching!</a></p>
        {self._create_image_html(self.images[0])}
        {self._create_signoff_html()}'''

        return MIMEText(msg_html, 'html', 'utf-8')

def send_email(email: Email, email_address: str, email_password: str) -> None:
    ''' Establishes connection and sends email `message` '''
    mailServer = smtplib.SMTP('smtp.gmail.com', 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login(email_address, email_password)
    mailServer.sendmail(email_address, email.recipient.email_address, email.email)
    mailServer.quit()