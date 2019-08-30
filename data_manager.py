import csv
import pandas_datareader as web

from preferences import Preferences
from models.financial_instruments import Stock


class DataManager(object):
    """
        This class is responsible for loading the data to operate on
    """

    def load_data(self, csv_path: str) -> list:
        """
            Creates a list of stock candidates that meet the requirements
            listed in the preferences to be placed into a portfolio
        """
        
        working_stocks = []
        print("Collecting bulk data...")

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            all_symbols = csv.reader(csvfile)
            for symbol in all_symbols:
                ticker = symbol[0]
                try:
                    data = web.DataReader(
                        name=ticker,
                        data_source='yahoo',
                        start=Preferences.BEGDATE,
                        end=Preferences.ENDDATE)
                except:
                    print("Couldn't find the ticker {0}".format(ticker))
                    continue
                else:
                    working_stocks.append(Stock(ticker, data))

                    print("successfully loaded {0}".format(ticker))

        print("Finished collecting bulk data")
        print("Successfully loaded {0} stocks".format(len(working_stocks)))

        return working_stocks