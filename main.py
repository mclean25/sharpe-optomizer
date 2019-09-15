import os
import pandas as pd
import sys
import math

from dateutil.relativedelta import relativedelta
from numpy import corrcoef, nan, array, ones
from datetime import datetime
from tqdm import tqdm

from preferences import Preferences
from data_loader import DataLoader
from universe_optimizer import UniverseOptimizer
from portfolio_optimizer import PortfolioOptimizer
from models.stocks import BulkStock, StockTimeFrame
from models.test_dates import TestDates
from models.financial_instruments import Portfolio, WeightedPortfolio
from models.matricies import Matricies

class TestFrame(object):

    def __init__(
        self,
        dates: TestDates,
        forecast_months : int,
        benchmark : BulkStock,
        stocks_universe: list):

        self.beg_date = dates.beg_date
        self.buy_date = dates.buy_date
        self.end_date = dates.end_date
        self.forecast_months = forecast_months

        self.stocks = self._create_test_stocks_list(stocks_universe)
        self.benchmark = StockTimeFrame(
            benchmark,
            dates.beg_date,
            dates.buy_date,
            dates.end_date
        )
        self.stocks_mapped = {s.stock.symbol: s for s in self.stocks}


    def run_test(self):
        universeOptimizer = UniverseOptimizer(self.stocks_mapped)
        matricies = Matricies(
            self.stocks_mapped,
            Preferences.RISK_FREE)                                                                  

        optimizer = UniverseOptimizer(self.stocks_mapped)
        portfolio_candidates = optimizer.create_portfolio_candidates(
            matricies,
            portfolio_size=Preferences.PORTFOLIO_SIZE)

        optimized_portfolios = []
        portfolio_optimizer = PortfolioOptimizer()

        print("optimizing portfolios")
        for portfolio in tqdm(portfolio_candidates):
            optimized_portfolio = portfolio_optimizer.optimize_portfolio(
                portfolio=portfolio,
                rf=Preferences.RISK_FREE,
                portfolio_size=Preferences.PORTFOLIO_SIZE,
                forecast_months=self.forecast_months)

            optimized_portfolio.calculate_post_returns(
                months_to_check = self.forecast_months,
                benchmark = self.benchmark)
                
            optimized_portfolios.append(optimized_portfolio)

        self.optimized_portfolios = sorted(
            optimized_portfolios,
            key=lambda x: x.portfolio_sharpe,
            reverse=True)


    def _create_test_stocks_list(self, stocks_universe) -> list:
        """
            Creates the list of stocks to be used in the test
        """
        test_stocks = []

        for stock in stocks_universe:
            if self._check_stock(stock):
                stock_time_frame = StockTimeFrame(
                    stock=stock,
                    beg_date=self.beg_date,
                    buy_date=self.buy_date,
                    end_date=self.end_date
                )

                if stock_time_frame.sharpe > 0:
                    test_stocks.append(stock_time_frame)

        return test_stocks


    def _check_stock(self, stock):
        """
            Checks that a stock has the required date for the test
            date ranges
        """
        if stock.bulk_data.index[0] <= self.beg_date:
            if stock.bulk_data.index[-1] > self.buy_date:
                return True
        
        return False



class Main(object):
    """
        Main hanlder for the program
    """

    def build_test_scenarios(self, csv_ticker_path: str) -> list:
        
        print("Will be loading tickers from the path {0}".format(csv_ticker_path))

        data_manager = DataLoader(os.path.join(os.getcwd(), 'cached_stock_data.sqlite'))
        stock_universe = data_manager.load_data(path_to_ticker_list)
        benchmark = data_manager.get_ticker_data('^GSPC')

        test_date_increment_months = 1
        test_historical_range_months = 24
        test_forecast_range_months = 12
        first_date = sorted([x.bulk_data.index[0] for x in stock_universe])[0]

        last_historic_test_date = datetime.today() - relativedelta(
            months=test_historical_range_months + test_forecast_range_months)

        if first_date > last_historic_test_date:
            raise Exception()

        # 30.5 for average days in a month
        number_of_tests = math.floor(((last_historic_test_date - first_date).days / 30.5) \
            / test_date_increment_months)

        print('Will be runnning {0}'.format(number_of_tests))

        test_frames = []

        for test_number in range(number_of_tests):
            beg_date = first_date + relativedelta(months=(test_number + 1) \
                * test_date_increment_months)
            buy_date = beg_date + relativedelta(months = test_historical_range_months)
            end_date = buy_date + relativedelta(months = test_forecast_range_months)

            test_frame = TestFrame(
                dates = TestDates(
                    beg_date = beg_date,
                    buy_date = buy_date,
                    end_date = end_date
                ),
                benchmark = benchmark,
                forecast_months = test_forecast_range_months,
                stocks_universe = stock_universe
            )

            test_frame.run_test()

            test_frames.add(test_frame)


    def _filter_stocks(self, stocks: list) -> list:
        buy_datetime = datetime.strptime(Preferences.PORTFOLIO_BUY_DATE, "%m/%d/%Y")
        allowed_stocks = []
        for stock in stocks:
            if stock.sharpe > 0 \
            and (stock.future_data_frame.index.max() - buy_datetime).days > 1:
                allowed_stocks.append(stock)
        
        return allowed_stocks


if __name__ == "__main__":
    print("Starting program")

    if len(sys.argv) > 1 and sys.argv[1] is not None :
        path_to_ticker_list = sys.argv[1]
    else:
        path_to_ticker_list = os.path.join(os.getcwd(), 'examples', 'symbols_short_list.csv')

    '''
        1. Get first available historical date (with n available stocks?)
        2. Each test iteration then becomes result of (1.) plus wanted historical test span plus 
            some arbitraty test iteration increment (eg: 1 month) to where there is sufficient forecasted
            (buy date + wanted length of forecasted data).
    '''
    test_date_increment_months = 1

    # get mode of beginning and end dates
    # take first n (percent of beginning dates)
    # that will be the first date to iterate on
    # keep iterating until END_DATE - wanted forcast date
            
    m = Main()

    m.build_test_scenarios(
        csv_ticker_path=path_to_ticker_list)

    print("Program finished")