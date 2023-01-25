import logging
import datetime as dt

class Date:
    @staticmethod
    def today() -> dt.date:
        return dt.date.today()

    @classmethod
    def last_week(cls) -> dt.date:
        return cls.today() - dt.timedelta(days=7)

class Logger:
    log_filename = 'log.txt'

    def __init__(self, frequency: str) -> None:
        self.frequency = frequency
        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename),
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