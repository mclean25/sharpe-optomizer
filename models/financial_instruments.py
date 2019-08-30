import numpy as np
import pandas as pd
import itertools as it

from preferences import Preferences


class Stock(object):
    """Object that holds a given stocks attributes"""

    percentage_change_col_identifier = 'Pct Change'
    adjusted_close_col_identifier = 'Adj Close'
 
    def __init__(self, symbol: str, data_frame: object):
        self.symbol = symbol

        self.historical_data_frame = self.calculate_adjusted_returns(
            data_frame[:Preferences.PORTFOLIO_BUY_DATE])

        self.forecasted_data_frame = self.calculate_adjusted_returns(
            data_frame[Preferences.PORTFOLIO_BUY_DATE:])

        self.calculate_metrics()


    def __str__(self):
        return self.symbol

    
    def calculate_adjusted_returns(self, data_frame: object) -> object:
        data_frame[self.percentage_change_col_identifier] = data_frame[
            self.adjusted_close_col_identifier].pct_change()
        
        return data_frame


    def calculate_metrics(self):
        self.mean = self.historical_data_frame[self.percentage_change_col_identifier].mean()
        self.risk = self.historical_data_frame[self.percentage_change_col_identifier].var()
        self.sharpe = self.risk ** 0.5



class Portfolio(object):

    def __init__(self, stocks: list, correlation_matrix: object):
        self.stocks = stocks
        self.correlations = self.build_correlation_combinations(correlation_matrix)
        self.array_data = PortfolioArrayData(stocks)


    def build_correlation_combinations(self, correlation_matrix):
        correlations = {}

        for combination in it.combinations(sorted([x.symbol for x in self.stocks]), 2):
            correlation = correlation_matrix[combination[0]][combination[1]]
            correlations[str(combination[0]) + str(combination[1])] = correlation

        return correlations


class PortfolioArrayData(object):
    """
        When using linear optimization, it's important to keep order of the inputs so we
        can tell what weights apply to which stocks from the output and that the risk and returns
        line up correctly (ie: the risk and returns have the same index in their respective lists)
    """

    def __init__(self, stocks: list):

        self.stocks = []
        self.returns = []
        self.stdevs = []
        
        for stock in stocks:
            self.stocks.append(stock.symbol)
            self.returns.append(stock.mean)
            self.stdevs.append(stock.risk)


    def add_weights(self):
        pass
        # TODO: add weights from linear optimization


class WeightedPortfolio(object):
    """
        Holds metrics for a given portfolio based on the given weights
        for the stocks in the portfolio
    """

    def __init__(self, portfolio, weights: list, risk_free: float):
        self.portfolio = portfolio
        self.weights = weights
        self.portfolio_return = self.calculate_portfolio_return()
        self.portfolio_risk = self.calculate_portfolio_risk()
        self.portfolio_sharpe = self.calculate_sharpe(risk_free)


    def calculate_portfolio_return(self):
        return sum(np.array(self.portfolio.array_data.returns) * self.weights) * 252


    def calculate_sharpe(self, rf):
        return (self.portfolio_return - rf) / self.portfolio_risk


    def calculate_portfolio_risk(self) -> float:
        variance = 0
        stocks_list = self.portfolio.stocks

        # calculating the first part of the variance formula
        for stock_index in range(len(stocks_list)):
            variance += (self.weights[stock_index]**2) \
                * (self.portfolio.array_data.stdevs[stock_index]**2)
        
        # calculating the second part of the variance formula
        for stock_combination in it.combinations(range(len(stocks_list)), 2):
            stock_a_name = stocks_list[stock_combination[0]].symbol
            stock_b_name = stocks_list[stock_combination[1]].symbol

            variance += 2 * self.weights[stock_combination[0]] * self.weights[stock_combination[1]] \
                * self.portfolio.correlations[stock_a_name + stock_b_name] \
                * self.portfolio.array_data.stdevs[stock_combination[0]] \
                * self.portfolio.array_data.stdevs[stock_combination[1]]

        return (variance * 252) ** 0.5

    def build_weighted_returns_data_series(self):
        """
            Creates a Pandas.DataFrame of the portfolio performance
        """

        # set the first item
        series = pd.DataFrame(
            {
                self.portfolio.stocks[0].symbol: self.portfolio.stocks[0] \
                    .forecasted_data_frame[Stock.percentage_change_col_identifier].copy()
            })

        series['{0} weight'.format(self.portfolio.stocks[0].symbol)] = self.weights[0]

        # handle the rest of the stocks in the portfolio
        for index, stock in enumerate(self.portfolio.stocks):
            if index > 0:
                series[stock.symbol] = stock.forecasted_data_frame[Stock.percentage_change_col_identifier]
                series['{0} weight'.format(stock.symbol)] = self.weights[index]
            
            series['{0} weighted returns'.format(stock.symbol)] = series[stock.symbol] \
                * series['{0} weight'.format(stock.symbol)]

        self.weighted_returns_data_series = series