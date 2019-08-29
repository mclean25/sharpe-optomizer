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
 
# y-axis = ranked sharpe
# x-axis = time after
# z-axis = returns
 
 
#Preferences:
#-------------------------------------------------------------------------------------------------------------------------
BEGDATE = "1/1/2017"
ENDDATE = datetime.datetime.now().date()
DATA_FREQUENCY = "d"
DEPTH_SCALE = 100
AVERAGE_SHARPE_CUTOFF = 0.0
WORKING_NUMBER = 100
PORTFOLIO_SIZE = 3
RISK_FREE = 0.005 / 252
COUNTRIES = ["Canada", "USA"]
 
HISTORICAL_RANGE = 300
EXANTE_RANGE = 300
TEST_AMOUNT = 1000
 
 
# Assumptions:
# --------------------------------------------------------------------------------------------------------------------------
"""
(1) If data for a given stock is unavailable at a certain day but available for another (due to holiday differences), that,
    day's data will be equal to the previous days.
(2) If data for a given stock is unavailable at the beginning data but available for another (due to holiday differences),
    this day's data is equal to the next day's data (backfill)
"""
 
class Stock(object):
    """Object that holds a given stocks attributes"""
 
    def __init__(self, name, symbol, exchange, country, data, mean, risk, sharpe, fdata):
        self.name = name
        self.symbol = symbol
        self.exchange = exchange
        self.country = country
        self.data = data
        self.mean = mean
        self.risk = risk
        self.sharpe = sharpe
        self.fdata = fdata
 
def create_stock(symbol, data, mean, risk, sharpe, fdata):
    """creates a stock object"""
    return Stock(name=symbol[1], symbol=symbol[0], exchange=symbol[2], country=symbol[3], data=data, mean=mean, risk=risk, sharpe=sharpe, fdata=fdata)
 
class Portfolio(object):
    """ Portfolio class """
 
    def __init__(self, stocks_list, stocks, weights, returns, sdeviations, correlations, mean, risk, sharpe, exante_array, exante_sum):
        self.stocks_list = stocks_list # we need this list because order does matter for the optimizer
        self.stocks = stocks # dictionary
        self.weights = weights # array
        self.returns = returns # list
        self.sdeviations = sdeviations # list
        self.correlations = correlations # dictionary
        self.mean = mean # float
        self.risk = risk # float
        self.sharpe = sharpe # float
        self.exante_array = exante_array
        self.exante_sum = exante_sum
 
def load_data(countries, beg_date, end_date):
    """ Creates a list of stock candidates that meet the requirements listed in the preferences to be placed into a portfolio"""
    working_stocks = []
    print("Collecting bulk data...")
    with open(open(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), 'allSymbolsSmall.csv')), 'r') as csvfile:
        all_symbols = csv.reader(csvfile)
        for symbol in all_symbols:
            try:
                if symbol[3] in countries:
                    # create the stock object after the stock has been verified to pass the minimum sharpe ratio to save time
                    try:
                        data = web.DataReader(symbol[0], 'yahoo', BEGDATE, ENDDATE)
                    except OSError:
                        continue
                    data['Pct Change'] = data['Adj Close'].pct_change()
                    df = data['Adj Close']
                    working_stocks.append(create_stock(symbol, data, None, None, None, None))
            except StopIteration:
                continue
    print("Finished collecting bulk data")
    print("size of bulk data: ", len(working_stocks))
    return list(set(working_stocks))
 
 
 
def create_standardized_timeseries(working_stocks):
    """ Creates a standardized timeseries to accurately compare correlation between proftolio candidates """
    seen_exchanges = set()
    #Sort the stocks by longest time series, then take the longest one from each exchange to create a timeseries standard
    print("length of working_stocks is: ", len(working_stocks))
    first = True
    standard = None
    working_stocks = sorted(working_stocks, key=lambda x: x.data.shape[0], reverse=True)
    print("stock exchanges used: ")
    for stock in working_stocks:
        if stock.exchange in seen_exchanges:
            continue
        else:
            #merge timeseries
            if first:
                standard = stock.data.ix[:, 0]
                first = False
            else:
                standard = pd.concat([standard,stock.data], axis=1).ix[:, 0]
                print("   ", stock.exchange)
                seen_exchanges.add(stock.exchange)
    if standard.any():
        return standard
    else:
        print("'standard' variable is None for some reason...")
        return
 
def standardize_stocks(working_stocks):
    """ standardizes all working_stocks to the standard time series"""
    standard = create_standardized_timeseries(working_stocks)
    print("Standard: ")
    print(standard)
    print("standard shape")
    print(standard.shape)
    standard_shape = standard.shape[0]
    for stock in working_stocks:
        stock.data= pd.concat([standard, stock.data], axis=1).iloc[:,[1,2,3,4,5,6]]
        stock.data= stock.data.ffill()
        stock.data= stock.data.bfill()
        # make a new percentage change column for computing correlation
        stock.data['pct_change2'] = stock.data['Adj Close'].pct_change()
    return working_stocks
 
def get_n_largest(matrix, n):
    return matrix.stack().drop_duplicates().sort_values(ascending=False).index[n]
 
def get_n_smallest(matrix, n):
    return matrix.stack().drop_duplicates().sort_values().index[n]
 
def compute_2stock_sharpe(stock_a, stock_b, correlation, rf):
    """returns a 2 stock sharpe value used to try and optimize portfolios"""
    weight_a = stock_a.sharpe / (stock_a.sharpe + stock_b.sharpe) # give the better ranked stock (individually) more weighting
    return ((stock_a.mean * weight_a) + (stock_b.mean * (1 - weight_a)) - rf) / (((stock_a.risk ** 2) * (weight_a ** 2)) + ((stock_b.risk ** 2) * ((1 - weight_a) ** 2)) + (2 * stock_a.risk * stock_b.risk * correlation * weight_a * (1 - weight_a)))
 
def stock_dataframe_name(stock):
    return stock.symbol
 
def create_matrix(working_stocks, rf):
    """ creates a sharpe matrix of the stocks in the current universe """
    # for stock in working_stocks:
    #     print(stock.name, stock.sharpe, stock.data.shape, stock.exchange)
    matrix = pd.concat([x.data['pct_change2'] for x in working_stocks], axis=1, keys=[stock_dataframe_name(x) for x in working_stocks]).corr()
    c_matrix, a_matrix = matrix.copy(), matrix.copy()
    # create a dictionary for easy stock lookup
    working_stocks_dict = {}
    for stock in working_stocks:
        working_stocks_dict[stock_dataframe_name(stock)] = stock
    for row in a_matrix:
        row_stock = working_stocks_dict[row]
        for col, v in a_matrix[row].iteritems():
            # print("row: ", row)
            # print("col: ", col)
            # print("matrix[row][col]", matrix[row][col])
            # print(" -------------------------------------------------------")
            # print("")
            col_stock = working_stocks_dict[col]
            if row_stock == col_stock:
                a_matrix[row][col] = nan
            else:
                a_matrix[row][col] = compute_2stock_sharpe(row_stock, col_stock, float(a_matrix[row][col]), rf)
    return c_matrix, a_matrix, working_stocks_dict
 
def check_settings(PORTFOLIO_SIZE):
    if PORTFOLIO_SIZE < 2:
        print("WARNING: PORTFOLIO SIZE SHOULD BE >=2")
 
def create_later_portfolio(index, portfolio, portfolio_size, loc_set, lead, a_matrix):
    """ Completes finding the portfolio stocks after the second stock """
    for cname, cval in a_matrix[lead].nlargest(index + portfolio_size)[index:].iteritems():
        if cname not in portfolio[0]:
            return cname
        else:
            continue
    print("couldn't find any stocks that arn't already in the portfolio in 'create_later_portfolio'")
 
def create_portfolio_candidates(c_matrix, a_matrix, portfolio_size):
    """ Returns a portfolio_size length list of stock names as a portfolio candidate """
    duplicate_counter = 0
    all_portfolios = []
    for depth_scaler in range(DEPTH_SCALE + 1):
        portfolio = [[],{}]
        if portfolio_size == 2:
            all_portfolios.append(list(get_n_largest(a_matrix, depth_scaler)))
        else:
            for first_index in range(2):
                # print("first_index: ", first_index)
                portfolio[0] = list(get_n_largest(a_matrix, depth_scaler))
                # print("portfolio", portfolio)
                lead = portfolio[0][first_index]
                # print('lead: ', lead)
                # we either start with the first or second stock of the highest rated pair in the matrix
                for loc_set in permutations(range(portfolio_size-2)):
                    # print("loc_set: ", loc_set)
                    loc_set = [x + depth_scaler for x in loc_set]
                    # print("modified loc_set: ", loc_set)
                    for index in loc_set:
                        # print("index: ", index)
                        old_lead = lead
                        lead = create_later_portfolio(index, portfolio, portfolio_size, loc_set, lead, a_matrix)
                        # print("new lead: ", lead)
                        portfolio[0].append(lead)
        sorted_port = sorted(portfolio[0])
        for comb in combinations(sorted_port, 2):
            # print("sorted_port: ", sorted_port)
            # print("comb: {0}".format(comb))
            stock_a, stock_b = comb[0], comb[1]
            portfolio[1][stock_a + stock_b] = c_matrix[stock_a][stock_b]
        # print("sorted_port: ", sorted_port)
        if True not in [sorted_port == seen_port for seen_port in all_portfolios]:
            # print("adding portfolio: {0}".format([sorted_port, portfolio[1]]))
            all_portfolios.append([sorted_port, portfolio[1]])
        else:
            duplicate_counter += 1
    print("Created {0} duplicate portfolios out of {1}, ({2}%)".format(duplicate_counter, duplicate_counter + len(all_portfolios), (duplicate_counter/(duplicate_counter + len(all_portfolios))*100)))
    return all_portfolios
 
def initialize_portfolios(portfolio_candidates, working_stocks_dict):
    all_portfolios = []
    for cand in portfolio_candidates:
        candidate, correlations = cand[0], cand[1]
        stocks = {}
        returns = []
        sdeviations = []
        for stock in candidate:
            stock_object = working_stocks_dict[stock_dataframe_name(working_stocks_dict[stock])]
            stocks[stock_dataframe_name(working_stocks_dict[stock])] = stock_object
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
 
 
 
l = datetime.datetime.now().date() - datetime.timedelta(EXANTE_RANGE)
f = datetime.datetime.now().date() - datetime.timedelta((EXANTE_RANGE + (TEST_AMOUNT / .70))) # 70% ~ percentage of working days/calendary days
 
 
bulk_data = load_data(COUNTRIES, f, l)
print("bulk_date: {0}".format(bulk_data))
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