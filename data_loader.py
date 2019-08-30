import csv
import pandas_datareader as web

from tqdm import tqdm

from preferences import Preferences
from models.financial_instruments import Stock


class DataLoader(object):
    """
        This class is responsible for loading the data to operate on
    """

    def load_data(self, csv_path: str) -> list:
        """
            Creates a list of stock candidates that meet the requirements
            listed in the preferences to be placed into a portfolio
        """
        
        working_stocks = []

        tickers_to_fetch = self._load_tickers_from_csv(csv_path)

        for ticker in tqdm(tickers_to_fetch):
            try:
                data = web.DataReader(
                    name=ticker,
                    data_source='yahoo',
                    start=Preferences.HISTORICAL_BEGINNING_DATE,
                    end=Preferences.ENDDATE)
            except:
                continue
            else:
                working_stocks.append(
                    Stock(
                        ticker,
                        data))

        print("Finished collecting bulk data")
        print("Successfully loaded {0} stocks".format(len(working_stocks)))

        return working_stocks


    def _load_tickers_from_csv(self, csv_path: str) -> list:
        
        tickers_to_fetch = []

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            all_symbols = csv.reader(csvfile)
            for symbol in all_symbols:
                tickers_to_fetch.append(symbol[0])

        return tickers_to_fetch