import datetime as dt
from siif_utils import PROD, TEST
from enum import Enum, auto

class Logger:

    def __init__(self, frequency: str, log_filename: str='log.txt', running_level: int=TEST) -> None:
        self.filename = log_filename
        self.frequency = frequency

    def begin(self) -> None:
        self._log(f'{self.frequency}\n=== Begin running at {str(dt.datetime.now())} ====\n')

    def success(self, name: str) -> None:
        self._log(f'Email to {name : >10} succeeded at {str(dt.datetime.now())}\n')

    def failure(self, name: str) -> None:
        self._log(f'Email to {name : >10} failed at {str(dt.datetime.now())}\n')

    def end(self) -> None:
        self._log(f'==== Completed emailing at {str(dt.datetime.now())} ====\n\n')

    def error(self, name: str) -> None:
        self._log(f'Error: {name : >10} {str(dt.datetime.now())}\n')

    def _log(self, message: str) -> None:
        # Print message so that user can see it
        print(message)

        with open(self.filename, 'a') as fp:
            fp.write(message)