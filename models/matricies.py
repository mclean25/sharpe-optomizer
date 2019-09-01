import pandas as pd
import numpy as np
import operator
import itertools as it

from tqdm import tqdm


class Matricies(object):
    """
        Houses matricies of the given stock universe
    """

    def __init__(self, stocks_mapped: dict, rf: float):
        print("building matricies")
        self.stocks_mapped = stocks_mapped

        stock_universe = list(stocks_mapped.values())

        self.correlation_matrix = self._build_correlation_matrix(
            stocks=stock_universe,
            rf=rf)

        self.sharpe_matrix = self._build_sharpe_matrix_new(
            correlation_matrix=self.correlation_matrix,
            rf=rf)

        self.ranked_pairings_descending_dict = self._stack_pairing_sharpes(
            sharpe_matrix = self.sharpe_matrix
        )

        self.flattened_matrix = self._flatten_sharpe_matrix(
            sharpe_by_ticker = self.ranked_pairings_descending_dict
        )


    def get_nlargest_sharpe_pairing(self, n) -> tuple:
        """
            Returns a tuple of the two stocks which have the `n` largest
            sharpe in the given matrix.
        """

        return self.flattened_matrix[n][0].split("/")


    def get_nlargest_sharpe_pairing_for_ticker(self, n, ticker) -> str:
        """
            Returns a tuple of the two stocks which have the `n` largest
            sharpe in the given matrix.
        """

        return self.ranked_pairings_descending_dict[ticker][n][0]


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
        print("building sharpe matrix")
        for row, v in tqdm(sharpe_matrix.iteritems()):
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


    def _build_sharpe_matrix_new(self, correlation_matrix, rf):
        permutations = it.permutations([x for x in self.stocks_mapped.keys()], 2)

        sharpes = {}

        for permutation in permutations:
            
            sharpe = None
            ticker_a = permutation[0]
            ticker_b = permutation[1]

            if ticker_a not in sharpes:
                sharpes[ticker_a] = {}

            # if the inverse pairing exists, it will be the same sharpe
            if ticker_b in sharpes and ticker_a in sharpes[ticker_b]:
                sharpe = sharpes[ticker_b][ticker_a]
            
            if sharpe is None:
                sharpe = self._compute_two_stock_sharpe(
                            stock_a=self.stocks_mapped[ticker_a],
                            stock_b=self.stocks_mapped[ticker_b],
                            correlation=float(correlation_matrix[ticker_a][ticker_b]),
                            rf=rf)

            sharpes[ticker_a][ticker_b] = sharpe

        return sharpes


    def _flatten_sharpe_matrix(self, sharpe_by_ticker: dict) -> dict:
        """
            Returns a list of tuples of all pairings and sharpe ratios in descending
            order based on their sharpe ratio.
        """
        seen_pairings = set()
        sharpes_flattened = []

        for ticker_a in sharpe_by_ticker.keys():
            pairings = sharpe_by_ticker[ticker_a]
            for pairing in pairings:
                ticker_b = pairing[0]
                
                # check that the inverse is not aLready been added
                if self._build_pairing_string(ticker_b, ticker_a) not in seen_pairings:
                    pairing_string = self._build_pairing_string(ticker_a, ticker_b)
                    sharpes_flattened.append(
                        (pairing_string, pairing[1])
                    )
                    seen_pairings.add(pairing_string)

        return sorted(sharpes_flattened, key=lambda x: x[1], reverse=True)


    def _build_pairing_string(self, ticker_a: str, ticker_b: str) -> str:
        return '{0}/{1}'.format(ticker_a, ticker_b)


    def _stack_pairing_sharpes(self, sharpe_matrix: dict) -> dict:
        """
            Returns a dict where the key is each ticker in the provided tickers list and the
            value is a list of tuples descending from highest sharpe ratio paring to lowest.
        """

        ranked_pairings = {}
        for ticker in sharpe_matrix.keys():
            ranked_pairings[ticker] = self._sort_sharpe_parings_for_ticker_desc(
                sharpe_matrix=sharpe_matrix,
                ticker=ticker)

        return ranked_pairings


    def _sort_sharpe_parings_for_ticker_desc(self, sharpe_matrix, ticker) -> list:
        """
            Sorts the sharpe ratio for each other pairing in the matrix for the given
            ticker in descending order (ie: the first item in the returned list will a tuple
            containing the ticker and sharpe ratio of the ticker that has the highest sharpe ratio
            with the provided ticker).
        """

        sorted_pairings = sorted(sharpe_matrix[ticker].items(), key=lambda kv: kv[1])
        sorted_pairings.reverse()
        return sorted_pairings


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
