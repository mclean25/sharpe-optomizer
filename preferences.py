import datetime

class Preferences(object):

    """
        This is the beginning of the historical analysis data used to compute the highest
        sharpe portfolios.
    """
    HISTORICAL_BEGINNING_DATE = "1/1/2017"

    """
        This is the end date of the historical analysis data. Any pricing data between this
        date and the `HISTORICAL_BEGINNING_DATE` will be used in evaluating portfolios with the largest
        sharpes.
    """
    PORTFOLIO_BUY_DATE = "1/1/2019"

    """
        This wil be the end of the future data. Any dates between `PORTFOLIO_BUY_DATE` and
        this date will be used to look at how well the portfolio performed after being bought.
    """
    ENDDATE = datetime.datetime.now().date()

    """
        This is the number of iterations deep the `universe_optimizer` will go in finding
        the `n` best sharpe ratio matches. For example, a value of `2` will tell the program
        to find the 1st, 2nd, and 3rd best sharpe pairing (two-stock) matchups for each stock
        in the portfolio.

        This value should be small for small stock universe lists but larger to increase the number of
        portfolios at the end of the program.
    """
    DEPTH_SCALE = 50

    """
        Number of stocks to include in a portfolio
    """
    PORTFOLIO_SIZE = 10

    """
        Risk free rate
    """
    RISK_FREE = 0.005 / 252