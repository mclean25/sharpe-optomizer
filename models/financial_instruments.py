import numpy as np
import pandas as pd
import itertools as it

from dateutil.relativedelta import relativedelta
from preferences import Preferences

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


    def calculate_post_returns(self, months_to_check: int):
        """
            Calculates the post returns for given portfolio
        """
        self._build_weighted_returns_data_series()
        self._calculate_cumulative_monthly_returns(months_to_check)


    def _build_weighted_returns_data_series(self):
        """
            Creates a Pandas.DataFrame of the portfolio performance
        """

        # set the first item
        series = pd.DataFrame(
            {
                self.portfolio.stocks[0].symbol: self.portfolio.stocks[0] \
                    .future_data_frame[Stock.percentage_change_col_identifier].copy()
            })

        series['Portfolio Cum. Returns'] = 0

        print('next portfolio')

        # handle the rest of the stocks in the portfolio
        for index, stock in enumerate(self.portfolio.stocks):
            adj_close_col_name = '{0} Adj close'.format(stock.symbol)
            stock_weight_name = '{0} Weight'.format(stock.symbol)
            adj_close_change_name = '{0} Pct Change'.format(stock.symbol)
            cumulative_returns_name = '{0} Cum. Returns'.format(stock.symbol)
            weighted_cum_returns_name = '{0} Weighted Cum. Returns'.format(stock.symbol)

            print('stock symbol: ' + stock.symbol)

            series[adj_close_col_name] = stock.future_data_frame[Stock.adjusted_close_col_identifier] \
                .fillna(method='ffill')

            series[adj_close_change_name] = series[adj_close_col_name].pct_change()

            series[stock_weight_name] = self.weights[index]

            series[cumulative_returns_name] = (1 + series[adj_close_change_name]).cumprod() - 1

            series[weighted_cum_returns_name] = series[cumulative_returns_name] * series[stock_weight_name]

            series['Portfolio Cum. Returns'] += series[weighted_cum_returns_name]

        self.weighted_returns_data_series = series

    
    def _calculate_cumulative_monthly_returns(self, months_to_check: int):
        """
            Calculates the cumulative monthly returns for the portfoio
        """

        months = [month + 1 for month in range(months_to_check)]
        
        first_date = self.weighted_returns_data_series.index[0]

        monthly_returns = {}
        for month in months:
            end_date = first_date + relativedelta(months=month)
            monthly_returns['Month {0}'.format(month)] = \
                self.weighted_returns_data_series[
                    first_date:end_date
                ]['Portfolio Cum. Returns'][-1]

        self.monthly_comulative_returns = monthly_returns