import os
import pandas as pd
import sys

from numpy import corrcoef, nan, array, ones

from preferences import Preferences
from data_loader import DataLoader
from universe_optimizer import UniverseOptimizer
from portfolio_optimizer import PortfolioOptimizer
from models.financial_instruments import Stock, Portfolio, WeightedPortfolio
from models.matricies import Matricies
 

class Main(object):
    """
        Main hanlder for the program
    """

    def get_best_sharpe_portfolios(csv_ticker_path: str) -> list:
        
        print("Will be loading tickers from the path {0}".format(csv_ticker_path))

        data_manager = DataLoader()
        stock_universe = data_manager.load_data(path_to_ticker_list)

        if len(stock_universe) < 2:
            print("Couln't load enough data")
            return None

        stocks_mapped = {s.symbol: s for s in stock_universe}

        universeOptimizer = UniverseOptimizer(stocks_mapped)
        matricies = Matricies(
            stocks_mapped,
            Preferences.RISK_FREE)

        optimizer = UniverseOptimizer(stocks_mapped)
        portfolios = optimizer.create_portfolio_candidates(
            matricies,
            portfolio_size=Preferences.PORTFOLIO_SIZE)

        optimized_portfolios = []
        portfolio_optimizer = PortfolioOptimizer()

        for portfolio in portfolios:
            optimized_portfolios.append(
                portfolio_optimizer.optimize_portfolio(
                    portfolio=portfolio,
                    rf=Preferences.RISK_FREE,
                    portfolio_size=Preferences.PORTFOLIO_SIZE))

        return optimized_portfolios


if __name__ is "__main__":
    """
        Pass in a csv where the first columnn is the list of tickers you would like to pull
        by running this application with `python main.py "C:\\my\\path\\to\\file.csv"`
    """

    print("Starting program")

    if len(sys.argv) > 1 and sys.argv[1] is not None :
        path_to_ticker_list = sys.argv[1]
    else:
        path_to_ticker_list = os.path.join(os.getcwd(), "examples\symbols_short_list.csv")
            
    Main()

    optimized_portfolios = Main.get_best_sharpe_portfolios(
        csv_ticker_path=path_to_ticker_list)

    print("Program finished")