import csv
import os
import urllib.request
import datetime
import heapq
import pandas_datareader as web
import pandas as pd

from itertools import permutations, combinations
from numpy import corrcoef, nan, array, ones
from scipy.optimize import minimize

from preferences import Preferences
from models.financial_instruments import Stock, Portfolio
 
# y-axis = ranked sharpe
# x-axis = time after
# z-axis = returns
 

class DataManager(object):
    """ This class is responsible for loading the data to operate on"""

    def load_data(self, csv_path: str) -> list:
        """ Creates a list of stock candidates that meet the requirements listed in the preferences to be placed into a portfolio"""
        
        working_stocks = []
        print("Collecting bulk data...")

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            all_symbols = csv.reader(csvfile)
            for symbol in all_symbols:
                ticker = symbol[0]
                # create the stock object after the stock has been verified to pass the minimum sharpe ratio to save time
                try:
                    data = web.DataReader(ticker, 'yahoo', Preferences.BEGDATE, Preferences.ENDDATE)
                except:
                    print("Couldn't find the ticker {0}".format(ticker))
                    continue
                else:
                    print("successfully loaded {0}".format(ticker))
                
                working_stocks.append(Stock(ticker, data))


        print("Finished collecting bulk data")
        print("Successfully loaded {0} stocks".format(len(working_stocks)))

        return working_stocks

# Assumptions:
# --------------------------------------------------------------------------------------------------------------------------
"""
(1) If data for a given stock is unavailable at a certain day but available for another
    (due to holiday differences), that, day's data will be equal to the previous days.
(2) If data for a given stock is unavailable at the beginning data but available for another (due to holiday differences),
    this day's data is equal to the next day's data (backfill)
"""

class Matricies(object):

    def __init__(self, stocks, rf):
        self.correlation_matrix = self._build_correlation_matrix(stocks, rf)
        self.sharpe_matrix = self._build_sharpe_matrix(self.correlation_matrix, rf)

        self.sharpe_matrix_stacked_descending = _stack_matrix(self.sharp_matrix, ascending=False)
        self.sharpe_matrix_stacked_ascending = _stack_matrix(self.sharp_matrix, ascending=True)
        self.correlation_matrix_stacked_descending = _stack_matrix(self.correlation_matrix, ascending=False)
        self.correlation_matrix_stacked_ascending = _stack_matrix(self.correlation_matrix, ascending=True)


    def get_n_largest_sharpe(self, n):
        return sharpe_matrix_stacked_descending.index(n)


    def get_n_largest_correlation(self, n):
        return correlation_matrix_stacked_descending.index(n)


    def get_n_smallest_sharpe(self, n):
        return sharpe_matrix_stacked_ascending.index(n)


    def get_n_smallest_correlation(self, n):
        return correlation_matrix_stacked_ascending.index(n)


    def _build_correlation_matrix(self, stocks, rf):
        return pd.concat(
            objs=[x.data_frame['Pct Change'] for x in stocks],
            axis=1,
            keys=[x.symbol for x in stocks]).corr()

    
    def _build_sharpe_matrix(self, correlation_matrix, rf):
        # first, set all values to nan
        sharpe_matrix = correlation_matrix.copy()
        sharpe_matrix[:] = nan

        for row in sharpe_matrix:
            row_stock = working_stocks_dict[row]
            for col, v in sharpe_matrix[row].iteritems():
                col_stock = working_stocks_dict[col]
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
        return matrix.stack().drop_duplicates().sort_values(ascending=isAscending)


    def _compute_two_stock_sharpe(self, stock_a, stock_b, correlation, rf):
        """returns a the sharpe portfolio of two combined stocks"""

        # arbitrarily giving more weighting proportionally to the higher rated stock
        weight_a = stock_a.sharpe / (stock_a.sharpe + stock_b.sharpe)

        combined_return = (stock_a.mean * weight_a) + (stock_b.mean * (1 - weight_a))
        combined_risk = ((stock_a.risk ** 2) * (weight_a ** 2)) + ((stock_b.risk ** 2) * ((1 - weight_a) ** 2)) + (2 * stock_a.risk * stock_b.risk * correlation * weight_a * (1 - weight_a))

        return (combined_return - rf) / combined_risk

    def _get_n_largest(matrix, n):
        return matrix.stack().drop_duplicates().sort_values(ascending=False).index[n]
    
    def _get_n_smallest(matrix, n):
        return matrix.stack().drop_duplicates().sort_values().index[n]



class UniverseOptimizer(object):
    """
        This class is designed to attempt to decrease the number of permutations required
        in order to solve for the highest sharpe ratio in a given universe.
    """

    def create_portfolio_candidates(matricies: Matricies, portfolio_size):
        """ Returns a portfolio_size length list of stock names as a portfolio candidate """
        duplicate_counter = 0
        all_portfolios = []

        # I think depth_scaler is how many of the n top sharpe parings to combine to make a portfolio
        for depth_scaler in range(Preferences.DEPTH_SCALE + 1):
            # list is stocks, dictionary is the sharpes for each matchup in the portfolio
            stocks = []
            if portfolio_size == 2:
                all_portfolios.append(matricies.get_n_largest_sharpe(depth_scaler))
            else:
                for first_index in range(2):
                    # print("first_index: ", first_index)
                    stocks = list(matricies.get_n_largest_sharpe(depth_scaler))
                    # print("portfolio", portfolio)
                    lead = stocks[first_index]
                    # print('lead: ', lead)
                    # we either start with the first or second stock of the highest rated pair in the matrix
                    for index_set in itertools.permutations(range(portfolio_size-2)):
                        # print("loc_set: ", loc_set)
                        index_set = [x + depth_scaler for x in index_set]
                        # print("modified loc_set: ", loc_set)
                        for index in index_set:
                            # print("index: ", index)
                            old_lead = lead
                            lead = create_later_portfolio(index, stocks, portfolio_size, index_set, lead, matricies.sharpe_matrix)
                            # print("new lead: ", lead)
                            stocks.append(lead)
            # print("sorted_port: ", sorted_port)
            # if True not in [sorted_port == seen_port for seen_port in all_portfolios]:
            #     # print("adding portfolio: {0}".format([sorted_port, portfolio[1]]))
            #     all_portfolios.append([sorted_port, portfolio[1]])
            # else:
            #     duplicate_counter += 1
        # print("Created {0} duplicate portfolios out of {1}, ({2}%)".format(duplicate_counter, duplicate_counter + len(all_portfolios), (duplicate_counter/(duplicate_counter + len(all_portfolios))*100)))
        return all_portfolios

        def create_later_portfolio(index, stocks, portfolio_size, index_set, lead, sharpe_matrix):
            """ Completes finding the portfolio stocks after the second stock """
            for cname, cval in sharpe_matrix[lead].nlargest(index + portfolio_size)[index:].iteritems():
                if cname not in portfolio:
                    return cname
                else:
                    continue
            print("couldn't find any stocks that arn't already in the portfolio in 'create_later_portfolio'")



class ToBeSorted(object):
    

    # def stock_dataframe_name(stock):
    #     return stock.symbol

    
    # def check_settings(PORTFOLIO_SIZE):
    #     if PORTFOLIO_SIZE < 2:
    #         print("WARNING: PORTFOLIO SIZE SHOULD BE >=2")
    
    def initialize_portfolios(portfolio_candidates, working_stocks_dict):
        all_portfolios = []
        for cand in portfolio_candidates:
            candidate, correlations = cand[0], cand[1]
            stocks = {}
            returns = []
            sdeviations = []
            for stock in candidate:
                stock_object = working_stocks_dict[working_stocks_dict[stock].name]
                stocks[working_stocks_dict[stock].name] = stock_object
                returns.append(stock_object.mean)
                sdeviations.append(stock_object.risk)
            all_portfolios.append(Portfolio(candidate, stocks, None, returns, sdeviations, correlations, None, None, None, None, None))
        return all_portfolios
    
    def portfolio_mean_var(portfolio, W):
        average = sum(array(portfolio.returns) * W) * 252
        # print("got average:{0} from returns: {1}".format(average, portfolio.returns))
        var1, var2 = 0, 0
        stocks_list = portfolio.stocks_list
        # print("weights in portfolio_mean_var: {0}".format(W))
        # print("sdeviations: {0}".format(portfolio.sdeviations))
        for i in range(len(stocks_list)):
            var1 += (W[i]**2) * (portfolio.sdeviations[i]**2)
        # print("Got var1 of :{0}".format(var1))
        for x in combinations(range(len(stocks_list)), 2):
            print(x)
            stock_a_name, stock_b_name = stocks_list[x[0]], stocks_list[x[1]]
            # print("correlation of stocks: {0},{1} will be {2}".format(stock_a_name, stock_b_name, portfolio.correlations[stock_a_name + stock_b_name]))
            var2 += 2 * W[x[0]] * W[x[1]] * portfolio.correlations[stock_a_name + stock_b_name] * portfolio.sdeviations[x[0]] * portfolio.sdeviations[x[1]]
            # print("got var2 of: {0}".format(var2))
        risk = ((var1 + var2) * 252) ** 0.5
        return average, risk
    
    def calculate_sharpe(portfolio, rf):
        # print("Calculating Sharpe with: MEAN:{0}, VAR:{1}, rf:{2}".format(portfolio.mean, portfolio.risk, rf))
        return (portfolio.mean - rf) / portfolio.risk
    
    def solve_weights(portfolio, rf, portfolio_size):
        def fitness(W, portfolio, rf):
            portfolio.mean, portfolio.risk = portfolio_mean_var(portfolio, W)
            util = calculate_sharpe(portfolio, rf)
            return 1/util
        W = ones([portfolio_size])/portfolio_size            # start optimization with equal weights
        b_ = [(0.,1.) for i in range(portfolio_size)]        # weights for boundaries between 0%..100%. No leverage, no shorting
        c_ = ({'type':'eq', 'fun': lambda W: sum(W) - 1.0})       # Sum of weights must be 100%
        optimized = minimize(fitness, W, (portfolio, rf), method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success:
            return None
        else:
            return optimized.x
    
    def calculate_portfolio_exante(port):
        arr = array([])
        for stock, i in enumerate(port.stocks_list):
            if arr:
                arr += array([stock.fdata['Cum Sum']] * port.weights[i])
            else:
                arr = array([stock.fdata['Cum Sum']] * port.weights[i])
        return arr
    
    
    def optimize_portfolio(portfolio, rf, portfolio_size):
        # print("Portfolio in optimize_portfolio", portfolio)
        W = solve_weights(portfolio, rf, portfolio_size)
        if W is None:
            W = ones([portfolio_size])/portfolio_size
        portfolio.weights = W
        portfolio.mean, portfolio.risk = portfolio_mean_var(portfolio, W)
        portfolio.sharpe = calculate_sharpe(portfolio, rf)
        portfolio.exante_array = calculate_portfolio_exante(portfolio)
        return portfolio
    
    # working_stocks = standardize_working_stocks(create_stock_universe(WORKING_NUMBER, AVERAGE_SHARPE_CUTOFF, RISK_FREE))
    # c_matrix, a_matrix, working_stocks_dict = create_matrix(working_stocks, RISK_FREE)
    # portfolios = initialize_portfolios(create_portfolio_candidates(c_matrix, a_matrix, PORTFOLIO_SIZE), working_stocks_dict)
    # portfolios = sorted([optimize_portfolio(x, RISK_FREE, PORTFOLIO_SIZE) for x in portfolios], key=lambda x: x.sharpe, reverse=True)
    
    
    def load_universe(bulk_data, working_number, beg_date, end_date, risk_free, exante_range):
        stock_universe = []
        for stock in bulk_data:
            bulk_data = stock.data
            stock.data = stock.data.loc[beg_date:end_date]
            ending_date_loc = stock.data.index.get_loc(end_date)
            fdata = bulk_data.iloc[ending_date_loc:ending_date_loc + exante_range]
            fdata['Pct Change'][0] = 0
            fdata['Cum Sum'] = fdata['Pct Change'].cumsum() 
            fmean = fdata['Pct Change'].mean()
            stock.fdata = fdata['Pct Change'].cumsum()
            stock.mean = stock.data['Pct Change'].mean()
            stock.risk = stock.data['Pct Change'].std()
            stock.sharpe = (stock.mean -  risk_free) / stock.risk
            if len(stock_universe) < working_number and len(stock_universe) != working_number - 1:
                stock_universe.append(stock)
            elif len(stock_universe) == working_number - 1:
                stock_universe.append(stock)
                stock_universe = sorted(stock_universe, key=lambda x: x.sharpe, reverse=True)
            else:
                if stock.sharpe > stock_universe[-1].sharpe:
                    del stock_universe[-1]
                    stock_universe.append(stock)
                    stock_universe = sorted(stock_universe, key=lambda x: x.sharpe, reverse=True)
                else:
                    continue
        return stock_universe
    
    
    
    # l = datetime.datetime.now().date() - datetime.timedelta(EXANTE_RANGE)
    # f = datetime.datetime.now().date() - datetime.timedelta((EXANTE_RANGE + (TEST_AMOUNT / .70))) # 70% ~ percentage of working days/calendary days
    
    
    # bulk_data = load_data(COUNTRIES, f, l)
    # print("bulk_date: {0}".format(bulk_data))


    # standardized_timeseries = create_standardized_timeseries(bulk_data)
    # # standardized_bulk_data = standardize_stocks(bulk_data)
    # mean_results_matrix = pd.DataFrame(index=list(range(10)), columns=list(range(EXANTE_RANGE)))
    # time_frames = [str(pd.to_datetime(x).date()) for x in standardized_timeseries.index]
    
    # if len(time_frames) < EXANTE_RANGE + HISTORICAL_RANGE:
    #     bump = EXANTE_RANGE + HISTORICAL_RANGE 
    
    # print("time_frames: {0}".format(time_frames))
    # beg_time_frames = time_frames[:-100]
    # end_time_frames = time_frames[100:]
    
    # all_ports = {}
    # results = {}
    
    # if len(beg_time_frames) == len(end_time_frames):
    #     for i in range(len(beg_time_frames)):
    #         print("Conducting Test: {0}".format(i))
    #         universe = load_universe(bulk_data, WORKING_NUMBER, beg_time_frames[i], end_time_frames[i], RISK_FREE)
    #         c_matrix, a_matrix, working_stocks_dict = create_matrix(working_stocks, RISK_FREE)
    #         portfolios = initialize_portfolios(create_portfolio_candidates(c_matrix, a_matrix, PORTFOLIO_SIZE), working_stocks_dict)
    #         portfolios = sorted([optimize_portfolio(x, RISK_FREE, PORTFOLIO_SIZE) for x in portfolios], key=lambda x: x.sharpe, reverse=True)
    #         rm = pd.DataFrame(index=list(range(10)), columns=list(range(EXANTE_RANGE)))
    #         for port, y in enumerate(portfolios):
    #             if y > 10:
    #                 break
    #             else:
    #                 for val, x in enumerate(port.exante_array):
    #                     rm.iloc[y][x] = val
    #         results[str(i)] = rm
    #     for p in results:
    #         for col, x in enumerate(results[p]):
    #             for val, y in enumerate(results[p][col]):
    #                 if pd.isnull(mean_results_matrix.iloc[y][x]):
    #                     mean_results_matrix.iloc[y][x] = val
    #                 else:
    #                     mean_results_matrix.iloc[y][x] += val
    #     mean_results_matrix = mean_results_matrix / len(beg_time_frames)
    #     print("MEAN RESULTS MATRIX")
    #     print(mean_results_matrix)
    #         # now we need to evaluate the performance of the portfolio afterwards
    # else:
    #     print("Problem with Dates... 'beg_time_frames' != 'end_time_frames'")


if __name__ is "__main__":
    print("hello")

    data_manager = DataManager()

    working_stocks = data_manager.load_data("C:\\Users\\alexa\\git\\sharpe-optimizer\\examples\\symbols_short_list.csv")
    working_stocks_by_sharpe = sorted(working_stocks, key=lambda x: x.sharpe, reverse=True)

    working_stocks_dict = {s.symbol: s for s in working_stocks_by_sharpe}

    universeOptimizer = UniverseOptimizer()
    matricies = Matricies(
        working_stocks,
        Preferences.RISK_FREE)

    print("done")