import datetime

class Preferences(object):
    BEGDATE = "1/1/2017"
    ENDDATE = datetime.datetime.now().date()
    DATA_FREQUENCY = "d"
    DEPTH_SCALE = 2
    AVERAGE_SHARPE_CUTOFF = 0.0
    WORKING_NUMBER = 100
    PORTFOLIO_SIZE = 3
    RISK_FREE = 0.005 / 252
    COUNTRIES = ["Canada", "USA"]
    
    HISTORICAL_RANGE = 300
    EXANTE_RANGE = 300
    TEST_AMOUNT = 1000