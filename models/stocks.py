class BulkStock(object):
    """
    Bulk stock data to be used for individual test
    """
    pct_change_col_id = 'Pct Change'
    adj_close_col_id = 'Adj Close'
    cumm_returns_col_id = 'Cum. Returns'
    days_diff_col_id = "Days diff"


    def __init__(self, symbol: str, bulk_data: object):
        self.symbol = symbol
        self.bulk_data = self._calculate_days_difference(
            self._calculate_change(bulk_data))


    def _calculate_days_difference(self, df):
        df[self.days_diff_col_id] = df.index
        df[self.days_diff_col_id] = (df[self.days_diff_col_id] - \
            df[self.days_diff_col_id].shift())

        return df

    
    def _calculate_change(self, df):
        df[BulkStock.percentage_change_col_identifier] = df[
            BulkStock.adj_close_col_id].pct_change()

        return df


    def __str__(self):
        return self.symbol


class StockTimeFrame(object):
    """
        Stock sliced for a given test time frame
    """

    @property
    def symbol(self):
        return self.stock.symbol

    
    @property
    def get_max_diff_historical_days(self):
        return self.historical_data_frame[BulkStock.days_diff_col_id].max()


    @property
    def get_max_diff_future_days(self):
        return self.historical_data_frame[BulkStock.days_diff_col_id].max()


    def __init__(self, stock: BulkStock, beg_date, buy_date, end_date):
        self.stock = stock
        self.beg_date = beg_date
        self.buy_date = buy_date
        self.end_date = end_date

        self.historical_data_frame = self.calculate_adjusted_returns(
            stock.bulk_data[beg_date:buy_date])

        self.future_data_frame = self.calculate_adjusted_returns(
            stock.bulk_data[buy_date:end_date])

        self.calculate_metrics()


    def __str__(self):
        return self.symbol


    def calculate_adjusted_returns(self, data: object) -> object:
        df = data.copy()

        df[BulkStock.cumulative_returns_col_identifier] = \
            (1 + df[BulkStock.pct_change_col_id]).cumprod() - 1

        if df.empty:
            print('is empty')
            pass
        
        return df


    def calculate_metrics(self):
        self.mean = self.historical_data_frame[BulkStock.pct_change_col_id].mean()
        self.risk = self.historical_data_frame[BulkStock.pct_change_col_id].var()
        self.sharpe = self.risk ** 0.5