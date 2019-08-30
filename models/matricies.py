import pandas as pd
import numpy as np


class Matricies(object):
    """
        Houses matricies of the given stock universe
    """

    def __init__(self, stocks_mapped: dict, rf: float):
        
        self.stocks_mapped = stocks_mapped

        stock_universe = list(stocks_mapped.values())

        self.correlation_matrix = self._build_correlation_matrix(
            stocks=stock_universe,
            rf=rf)

        self.sharpe_matrix = self._build_sharpe_matrix(
            correlation_matrix=self.correlation_matrix,
            rf=rf)

        self.sharpe_matrix_stacked_descending = self._stack_matrix(
            matrix=self.sharpe_matrix,
            isAscending=False)


    def get_nlargest_sharpe_pairing(self, n) -> tuple:
        """
            Returns a tuple of the two stocks which have the `n` largest
            sharpe in the given matrix.
        """

        return self.sharpe_matrix_stacked_descending.index[n]


    def _build_correlation_matrix(self, stocks: list, rf: float):
        """
            Builds a matrix of the given stock universe with the correrlation
            of each stock against every other stock.
        """

        return pd.concat(
            objs=[x.historical_data_frame['Pct Change'] for x in stocks],
            axis=1,
            keys=[x.symbol for x in stocks]).corr()

    
    def _build_sharpe_matrix(self, correlation_matrix, rf):
        """
            Builds a matrix of the given stock universe with the sharpe ratio
            of each stock compared to each other stock as if they made a two
            stock portfolio with equal weights.
        """

        sharpe_matrix = correlation_matrix.copy()

        # first, set all values to nan
        sharpe_matrix[:] = np.nan

        for row in sharpe_matrix:
            row_stock = self.stocks_mapped[row]
            for col, v in sharpe_matrix[row].iteritems():
                col_stock = self.stocks_mapped[col]
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
        """
            Stacks the given matrix, which is useful for ranking
        """

        return matrix.stack().drop_duplicates().sort_values(ascending=isAscending)


    def _compute_two_stock_sharpe(self, stock_a, stock_b, correlation, rf) -> float:
        """
            Returns the sharpe ratio of the two given stocks as if the two stocks
            solely made up the portfolio and both shared equal weighting in the portfolio
        """

        # arbitrarily giving more weighting proportionally to the higher rated stock
        weight_a = stock_a.sharpe / (stock_a.sharpe + stock_b.sharpe)

        combined_return = (stock_a.mean * weight_a) + (stock_b.mean * (1 - weight_a))

        combined_risk = ((stock_a.risk ** 2) * (weight_a ** 2)) \
            + ((stock_b.risk ** 2) * ((1 - weight_a) ** 2)) \
            + (2 * stock_a.risk * stock_b.risk * correlation * weight_a * (1 - weight_a))

        return (combined_return - rf) / combined_risk
