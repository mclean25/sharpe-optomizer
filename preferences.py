import datetime

class Preferences(object):
    HISTORICAL_BEGINNING_DATE = "1/1/2017"
    # this is the date we buy the portfolio and also
    # the day after we finish evaluating sharpe ratio data
    PORTFOLIO_BUY_DATE = "1/1/2019"
    ENDDATE = datetime.datetime.now().date()
    DEPTH_SCALE = 2
    PORTFOLIO_SIZE = 3
    RISK_FREE = 0.005 / 252