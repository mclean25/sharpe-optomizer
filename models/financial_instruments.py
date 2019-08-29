import numpy as np
import itertools as it


class Stock(object):
    """Object that holds a given stocks attributes"""

    percentage_change_col_identifier = 'Pct Change'
    adjusted_close_col_identifier = 'Adj Close'
 
    def __init__(self, symbol: str, data_frame: object):
        self.symbol = symbol
        self.data_frame = self.calculate_adjusted_returns(data_frame)
        self.calculate_metrics()

    def __str__(self):
        return self.symbol

    
    def calculate_adjusted_returns(self, data_frame: object) -> object:
        data_frame[self.percentage_change_col_identifier] = data_frame[
            self.adjusted_close_col_identifier].pct_change()
        
        return data_frame


    def calculate_metrics(self):
        self.mean = self.data_frame[self.percentage_change_col_identifier].mean()
        self.risk = self.data_frame[self.percentage_change_col_identifier].var()
        self.sharpe = self.risk ** 0.5

class Portfolio(object):

    def __init__(self, stocks: list, correlation_matrix: object):
        self.stocks = stocks
        self.correlations = build_correlation_combinations(correlation_matrix)

    def build_correlation_combinations(self, correlation_matrix):
        correlations = {}

        for combination in it.combinations(sorted(self.stocks), 2):
            correlation = correlation_matrix[combination[0]][combination[1]]
            correlations[str(combination[0]) + str(combination[1])] = correlation

        return correlations


# class Portfolio(object):
#     """ Portfolio class """
 
#     def __init__(
#             self,
#             stocks_list,
#             stocks,
#             weights,
#             returns,
#             sdeviations,
#             correlations,
#             mean,
#             risk,
#             sharpe,
#             exante_array,
#             exante_sum):
#         self.stocks_list = stocks_list # we need this list because order does matter for the optimizer
#         self.stocks = stocks # dictionary
#         self.weights = weights # array
#         self.returns = returns # list
#         self.sdeviations = sdeviations # list
#         self.correlations = correlations # dictionary
#         self.mean = mean # float
#         self.risk = risk # float
#         self.sharpe = sharpe # float
#         self.exante_array = exante_array
#         self.exante_sum = exante_sum