import csv
import datetime
import sqlite3 as sq
import pandas as pd
import pandas_datareader as web

from tqdm import tqdm

from preferences import Preferences
from models.financial_instruments import Stock


class DataLoader(object):
    """
        This class is responsible for loading the data to operate on
    """

    def __init__(self, path_to_database: str):
        self.conn = sq.connect(path_to_database)


    def load_data(self, csv_path: str) -> list:
        """
            Creates a list of stock candidates that meet the requirements
            listed in the preferences to be placed into a portfolio
        """

        # Will create a connection if one does not exist
        
        working_stocks = []

        tickers_to_fetch = self._load_tickers_from_csv(csv_path)
        todays_date = datetime.datetime.today()

        print("Loading stock data")
        for ticker in tqdm(tickers_to_fetch):
            # try to load from the db first

            data = self._load_from_sql(
                ticker=ticker
            )

            # check that the last date is < n days away from the current date.
            # Idea is to roughly handle long weekends.
            if data is None or (datetime.datetime.today() - data.index[-1]).days > 4:
                try:
                    data = web.DataReader(
                        name=ticker,
                        data_source='yahoo')
                except:
                    continue
                else:
                    self._add_data_to_db(
                        data=data,
                        ticker=ticker
                    )

            if data is not None:
                working_stocks.append(Stock(ticker, data))


        print("Successfully loaded {0} stocks".format(len(working_stocks)))

        return working_stocks


    def _load_tickers_from_csv(self, csv_path: str) -> list:
        
        tickers_to_fetch = []

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            all_symbols = csv.reader(csvfile)
            for symbol in all_symbols:
                tickers_to_fetch.append(symbol[0])

        return tickers_to_fetch

    
    def _load_from_sql(self, ticker: str) -> list:

        data = None

        try:
            data = pd.read_sql('select * from {0}'.format(ticker), self.conn)
        except pd.io.sql.DatabaseError:
            # ticker does not exist
            pass

        if not data is None:
            # set the index to the date
            data = data.set_index(pd.to_datetime(data['Date'], format="%Y-%m-%d"))
            data = data.drop(columns=['Date'])

        return data


    def _add_data_to_db(self, data: pd.DataFrame, ticker: str):
        data.to_sql(ticker, self.conn, if_exists='replace', index=True)