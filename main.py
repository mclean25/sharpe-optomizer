import csv
import os
import urllib.request
import datetime
import heapq
import pandas_datareader as web
import pandas as pd
import itertools

from numpy import corrcoef, nan, array, ones
from scipy.optimize import minimize

from preferences import Preferences
from models.financial_instruments import Stock, Portfolio, WeightedPortfolio
 

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
                # create the stock object after the stock has been verified to pass the minimum sharpe ratio to save time
                try:
                    data = web.DataReader(ticker, 'yahoo', Preferences.BEGDATE, Preferences.ENDDATE)
                except:
                    print("Couldn't find the ticker {0}".format(ticker))
                    continue
                else:
                    working_stocks.append(Stock(ticker, data))

                    print("successfully loaded {0}".format(ticker))

        print("Finished collecting bulk data")
        print("Successfully loaded {0} stocks".format(len(working_stocks)))

        return working_stocks


class Matricies(object):

    def __init__(self, stocks, rf):
        self.correlation_matrix = self._build_correlation_matrix(stocks, rf)
        self.sharpe_matrix = self._build_sharpe_matrix(self.correlation_matrix, rf)

        self.sharpe_matrix_stacked_descending = self._stack_matrix(
            self.sharpe_matrix,
            isAscending=False)

        self.sharpe_matrix_stacked_ascending = self._stack_matrix(
            self.sharpe_matrix,
            isAscending=True)
        
        # self.correlation_matrix_stacked_descending = self._stack_matrix(self.correlation_matrix, isAscending=False)
        # self.correlation_matrix_stacked_ascending = self._stack_matrix(self.correlation_matrix, isAscending=True)


    def get_n_largest_sharpe_pairing(self, n):
        return self.sharpe_matrix_stacked_descending.index[n]


    def get_n_smallest_sharpe_pairing(self, n):
        return self.sharpe_matrix_stacked_ascending.index[n]


    def _build_correlation_matrix(self, stocks, rf):
        return pd.concat(
            objs=[x.data_frame['Pct Change'] for x in stocks],
            axis=1,
            keys=[x.symbol for x in stocks]).corr()

    
    def _build_sharpe_matrix(self, correlation_matrix, rf):
        # first, set all values to nan
        sharpe_matrix = correlation_matrix.copy()
        sharpe_matrix[:] = nan

        for row in sharpe_matrix:
            row_stock = stocks_mapped[row]
            for col, v in sharpe_matrix[row].iteritems():
                col_stock = stocks_mapped[col]
                # if the inverse index in the matrix is already calculated, then
                # we don't need to calculate again.
                if not pd.isna(sharpe_matrix[col][row]):
                    sharpe_matrix[row][col] = sharpe_matrix[col][row]
                elif row_stock != col_stock:
                    sharpe_matrix[row][col] = self._compute_two_stock_sharpe(
                        stock_a=row_stock,
                        stock_b=col_stock,
                        correlation=float(correlation_matrix[row][col]),
                        rf=rf)

        return sharpe_matrix


    def _stack_matrix(self, matrix, isAscending):
        return matrix.stack().drop_duplicates().sort_values(ascending=isAscending)


    def _compute_two_stock_sharpe(self, stock_a, stock_b, correlation, rf):
        """returns a the sharpe portfolio of two combined stocks"""

        # arbitrarily giving more weighting proportionally to the higher rated stock
        weight_a = stock_a.sharpe / (stock_a.sharpe + stock_b.sharpe)

        combined_return = (stock_a.mean * weight_a) + (stock_b.mean * (1 - weight_a))

        combined_risk = ((stock_a.risk ** 2) * (weight_a ** 2)) \
            + ((stock_b.risk ** 2) * ((1 - weight_a) ** 2)) \
            + (2 * stock_a.risk * stock_b.risk * correlation * weight_a * (1 - weight_a))

        return (combined_return - rf) / combined_risk



class UniverseOptimizer(object):
    """
        This class is designed to attempt to decrease the number of permutations required
        in order to solve for the highest sharpe ratio in a given universe.
    """

    def __init__(self, stocks_mapped: dict):
        self.stocks_mapped = stocks_mapped

    def create_portfolio_candidates(self, matricies, portfolio_size: int):
        """ Returns a portfolio_size length list of stock names as a portfolio candidate """
        duplicate_counter = 0
        all_portfolios = []

        # I think depth_scaler is how many of the n top sharpe parings to combine to make a portfolio
        for depth_scaler in range(Preferences.DEPTH_SCALE + 1):
            if portfolio_size == 2:
                all_portfolios.append(sorted(list(matricies.get_n_largest_sharpe(depth_scaler))))
            else:
                portfolio_stocks = []
                # this will become a tuple of two ticker strings
                # the "lead" stock is the stock we follow to get the next `n` best sharpe ratio.
                # this is how we follow the sharpe matrix to build a list of portfolio candidates.
                first_two_stocks = matricies.get_n_largest_sharpe_pairing(depth_scaler)
                for lead_ticker in first_two_stocks:
                    portfolio_stocks = list(first_two_stocks)
                    # iterate every permutation for the remainder of required stocks in the portfolio
                    for index_set in itertools.permutations(range(portfolio_size-2)):
                        # scale these indexes up by the depth scalar
                        scaled_index_set = [x + depth_scaler for x in index_set]
                        for index in scaled_index_set:
                            
                            lead_ticker = self.find_next_ticker_candidate(
                                index=index,
                                current_stocks=portfolio_stocks,
                                lead_ticker=lead_ticker,
                                matricies=matricies)

                            portfolio_stocks.append(lead_ticker)
                            
                    all_portfolios.append(sorted(portfolio_stocks))

        # clean out all of the portfolios that contain the same stocks
        return self.create_unique_portfolios(all_portfolios, matricies)

    
    def create_unique_portfolios(self, portfolio_candidates: list, matricies) -> list:
        seen_portfolio_candidates = []
        portfolios = []
        for portfolio_candidate in portfolio_candidates:
            if portfolio_candidate not in seen_portfolio_candidates:
                seen_portfolio_candidates.append(portfolio_candidate)
                stocks = [self.stocks_mapped[x] for x in portfolio_candidate]
                portfolios.append(
                    Portfolio(
                        stocks=stocks,
                        correlation_matrix=matricies.correlation_matrix))

        return portfolios


    def find_next_ticker_candidate(self, index, current_stocks, lead_ticker, matricies):
        """
            Finds the next ticker to include in the portfolio given a "lead" ticker to follow in
            the sharpe matrix.
        """

        for ticker in matricies.sharpe_matrix[lead_ticker].sort_values(ascending=False).index:
            if ticker not in current_stocks:
                return ticker
            else:
                continue

        raise Exception()

        print("couldn't find any stocks that arn't already in the portfolio in 'create_later_portfolio'")


class PortfolioOptimizer(object):

    def optimize_portfolio(self, portfolio, rf: float, portfolio_size: int):
        weights = self.solve_weights(portfolio, rf, portfolio_size)

        if weights is None:
            weights = ones([portfolio_size])/portfolio_size

        return WeightedPortfolio(
            portfolio=portfolio,
            weights=weights,
            risk_free=rf)
    

    def solve_weights(self, portfolio, rf, portfolio_size):
        def fitness(weights, portfolio, rf):
            weighted_portfolio = WeightedPortfolio(
                portfolio=portfolio,
                weights=weights,
                risk_free=rf)
            utility = weighted_portfolio.portfolio_sharpe
            return 1/utility

        # start optimization with equal weights
        weights = ones([portfolio_size])/portfolio_size
        # weights must be between 0 - 100%, assuming no shorting or leveraging
        bounds = [(0.,1.) for i in range(portfolio_size)]
        # Sum of weights must be 100%
        constraints = ({'type':'eq', 'fun': lambda weights: sum(weights) - 1.0})       
        optimized = minimize(fitness, weights, (portfolio, rf), method='SLSQP', constraints=constraints, bounds=bounds)
        if not optimized.success:
            return None
        else:
            return optimized.x
    

if __name__ is "__main__":
    print("hello")

    data_manager = DataManager()

    working_stocks = data_manager.load_data("C:\\Users\\alexa\\git\\sharpe-optimizer\\examples\\symbols_short_list.csv")
    working_stocks_by_sharpe = sorted(working_stocks, key=lambda x: x.sharpe, reverse=True)

    stocks_mapped = {s.symbol: s for s in working_stocks_by_sharpe}

    universeOptimizer = UniverseOptimizer(stocks_mapped)
    matricies = Matricies(
        working_stocks,
        Preferences.RISK_FREE)

    # 'Special Sauce' on attempting to get around the NP problem
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

    print("done")