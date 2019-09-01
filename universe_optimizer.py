import itertools

from tqdm import tqdm

from preferences import Preferences
from models.financial_instruments import Portfolio

class UniverseOptimizer(object):
    """
        This class is designed to attempt to decrease the number of permutations required
        in order to solve for the highest sharpe ratio in a given universe.

        Sort of the "special sauce" of the program
    """

    def __init__(self, stocks_mapped: dict):
        self.stocks_mapped = stocks_mapped

    def create_portfolio_candidates(self, matricies, portfolio_size: int):
        """ Returns a portfolio_size length list of stock names as a portfolio candidate """
        duplicate_counter = 0
        all_portfolios = []

        print("building portfolio candidates")
        # I think depth_scaler is how many of the n top sharpe parings to combine to make a portfolio
        for depth_scaler in tqdm(range(Preferences.DEPTH_SCALE + 1)):
            if portfolio_size == 2:
                all_portfolios.append(sorted(list(matricies.get_nlargest_sharpe_pairing(depth_scaler))))
            else:
                portfolio_stocks = []
                # this will become a tuple of two ticker strings
                # the "lead" stock is the stock we follow to get the next `n` best sharpe ratio.
                # this is how we follow the sharpe matrix to build a list of portfolio candidates.
                first_two_stocks = matricies.get_nlargest_sharpe_pairing(depth_scaler)
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
                        portfolio_stocks = list(first_two_stocks)

        # clean out all of the portfolios that contain the same stocks
        return self.create_unique_portfolios(all_portfolios, matricies)

    
    def create_unique_portfolios(self, portfolio_candidates: list, matricies) -> list:
        seen_portfolio_candidates = []
        portfolios = []
        print("cleaning portfolios")
        for portfolio_candidate in tqdm(portfolio_candidates):
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
        #TODO: should this use depth scale as well?
        for pairing_sharpe_tuple in matricies.ranked_pairings_descending_dict[lead_ticker]:
            ticker = pairing_sharpe_tuple[0]
            if ticker not in current_stocks:
                return ticker
            else:
                continue

        raise Exception()

        print("couldn't find any stocks that arn't already in the portfolio in 'create_later_portfolio'")