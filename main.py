import os
import pandas as pd
import sys

from numpy import corrcoef, nan, array, ones

from datetime import datetime

from preferences import Preferences
from tqdm import tqdm
from data_loader import DataLoader
from universe_optimizer import UniverseOptimizer
from portfolio_optimizer import PortfolioOptimizer
from models.financial_instruments import Stock, Portfolio, WeightedPortfolio
from models.matricies import Matricies
 
class Main(object):
    """
        Main hanlder for the program
    """

    def get_best_sharpe_portfolios(self, csv_ticker_path: str) -> list:
        
        print("Will be loading tickers from the path {0}".format(csv_ticker_path))

        data_manager = DataLoader(os.path.join(os.getcwd(), 'cached_stock_data.sqlite'))
        stock_universe = data_manager.load_data(path_to_ticker_list)

        print('stocks count before filter {0}'.format(len(stock_universe)))
        stock_universe = self._filter_stocks(stock_universe)
        print('stocks count after filter {0}'.format(len(stock_universe)))

        if len(stock_universe) < 2:
            print("Couldn't load enough data")
            return None

        stocks_mapped = {s.symbol: s for s in stock_universe}

        universeOptimizer = UniverseOptimizer(stocks_mapped)
        matricies = Matricies(
            stocks_mapped,
            Preferences.RISK_FREE)

        optimizer = UniverseOptimizer(stocks_mapped)
        portfolio_candidates = optimizer.create_portfolio_candidates(
            matricies,
            portfolio_size=Preferences.PORTFOLIO_SIZE)

        optimized_portfolios = []
        portfolio_optimizer = PortfolioOptimizer()

        print("optimizing portfolios")
        for portfolio in tqdm(portfolio_candidates):
            optimized_portfolios.append(
                portfolio_optimizer.optimize_portfolio(
                    portfolio=portfolio,
                    rf=Preferences.RISK_FREE,
                    portfolio_size=Preferences.PORTFOLIO_SIZE))

        return optimized_portfolios


    def _filter_stocks(self, stocks: list) -> list:
        buy_datetime = datetime.strptime(Preferences.PORTFOLIO_BUY_DATE, "%m/%d/%Y")
        allowed_stocks = []
        for stock in stocks:
            if stock.sharpe > 0 \
            and (stock.forecasted_data_frame.index.max() - buy_datetime).days > 1:
                allowed_stocks.append(stock)
        
        return allowed_stocks



if __name__ == "__main__":
    print("Starting program")

    if len(sys.argv) > 1 and sys.argv[1] is not None :
        path_to_ticker_list = sys.argv[1]
    else:
        path_to_ticker_list = os.path.join(os.getcwd(), "examples\symbols_short_list.csv")
            
    m = Main()

    optimized_portfolios = m.get_best_sharpe_portfolios(
        csv_ticker_path=path_to_ticker_list)

    optimized_portfolios = sorted(optimized_portfolios, key=lambda x: x.portfolio_sharpe, reverse=True)

    print("Program finished")