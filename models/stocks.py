class BulkStock(object):
    """
    Bulk stock data to be used for individual test
    """
    percentage_change_col_identifier = 'Pct Change'
    adjusted_close_col_identifier = 'Adj Close'
    cumulative_returns_col_identifier = 'Cum. Returns'

    def __init__(self, symbol: str, bulk_data: object):
        self.symbol = symbol
        self.bulk_data = bulk_data


    def __str__(self):
        return self.symbol


class StockTimeFrame(object):
    """
        Stock sliced for a given test time frame
    """

    @property
    def symbol(self):
        return self.stock.symbol

    def __init__(self, stock: BulkStock, beg_date, buy_date, end_date):
        self.stock = stock

        self.historical_data_frame = self.calculate_adjusted_returns(
            stock.bulk_data[beg_date:buy_date].copy())

        self.future_data_frame = self.calculate_adjusted_returns(
            stock.bulk_data[buy_date:end_date].copy())

        self.calculate_metrics()


    def __str__(self):
        return self.symbol


    def calculate_adjusted_returns(self, data: object) -> object:
        data[BulkStock.percentage_change_col_identifier] = data[
            BulkStock.adjusted_close_col_identifier].pct_change()

        data[BulkStock.cumulative_returns_col_identifier] = \
            (1 + data[BulkStock.percentage_change_col_identifier]).cumprod() - 1
        
        return data


    def calculate_metrics(self):
        self.mean = self.historical_data_frame[BulkStock.percentage_change_col_identifier].mean()
        self.risk = self.historical_data_frame[BulkStock.percentage_change_col_identifier].var()
        self.sharpe = self.risk ** 0.5