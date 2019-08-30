import datetime

class Preferences(object):
    BEGDATE = "1/1/2017"
    ENDDATE = datetime.datetime.now().date()
    DEPTH_SCALE = 2
    PORTFOLIO_SIZE = 3
    RISK_FREE = 0.005 / 252