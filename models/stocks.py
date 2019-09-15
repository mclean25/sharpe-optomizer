class Stock(object):
    """
    Bulk stock data to be used for individual test
    """

    percentage_change_col_identifier = 'Pct Change'
    adjusted_close_col_identifier = 'Adj Close'

    @property
    def get_first_date(self):
        return self.data_frame.index[0]


    def __init__(self, symbol: str, data_frame: object):
        self.symbol = symbol
        self.date_frame = data_frame


    def __str__(self):
        return self.symbol



class StockTimeFrame(object):
    """
        Stock sliced for a given test time frame
    """

    def __init__(self, stock: Stock, beg_date, buy_date, end_date):
        self.stock = stock

        self.historical_data_frame = self.calculate_adjusted_returns(
            stock.data_frame[beg_date:buy_date])

        self.future_data_frame = self.calculate_adjusted_returns(
            stock.data_frame[buy_date:end_date])

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