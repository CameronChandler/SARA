import logging
from enum import Enum, auto
import datetime as dt

class Date:
    @staticmethod
    def today() -> dt.date:
        return dt.date.today()

    @classmethod
    def last_week(cls) -> dt.date:
        return cls.today() - dt.timedelta(days=7)

class RunningLevel(Enum):
    PROD = auto()
    TEST = auto()

class Logger:
    log_filename = {RunningLevel.PROD: 'log.txt', RunningLevel.TEST: 'test_log.txt'}

    def __init__(self, frequency: str, running_level: RunningLevel=RunningLevel.TEST) -> None:
        self.frequency = frequency
        self.running_level = running_level
        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename[self.running_level]),
                logging.StreamHandler()
            ],
            level=logging.INFO
        )

    def begin(self) -> None:
        logging.info(f'==== Begin running {self.frequency} ====')

    def success(self, name: str) -> None:
        logging.info(f'Email to {name : >10} succeeded')

    def failure(self, name: str) -> None:
        logging.error(f'Email to {name : >10} failed')

    def end(self) -> None:
        logging.info(f'==== Completed emailing ====\n')

    def error(self, name: str) -> None:
        logging.error(f'Error: {name : >10}')