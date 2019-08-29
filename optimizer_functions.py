import numpy 
from numpy import ones, linspace, transpose
from numpy.linalg import inv, pinv
import scipy.optimize
import csv, os, platform, json, time, locale, sys, bisect, math, os.path, itertools, re
from preferences import *
from functions import *
 
class DropStock(Exception):
    pass
 
def get_data(bulk_data, coordinate):
    names = []
    rates = []
    sigma = []
    means = []
    exchanges = []
    for spot in coordinate:
        names.append(bulk_data[spot]['name'])
        rates.append(bulk_data[spot]['rates'])
        sigma.append(bulk_data[spot]['std_returns'])
        means.append(bulk_data[spot]['avg_returns'])
        exchanges.append(bulk_data[spot]['exchange'])
    weights = numpy.array(numpy.random.dirichlet(numpy.ones(len(names))*1000.,size=1)) # random starter weights
    rates = numpy.array(rates)
    sigma = numpy.array(sigma)
    means = numpy.array(means)
    C = numpy.cov(rates)
    Corr = numpy.corrcoef(rates)
    # Annualizing:
    # C = C * 251
    # Means are already annualized!
    return names, weights, means, C, Corr, sigma, exchanges
 
def port_mean(W,R):
    try:
       return sum(W*R)
    except TypeError:
        print "W: ", W
        print "R: ", R
 
def port_var(n, W, Corr, sigma):
    var1 = []
    var2 = []
    for c in range(n):
        try:
            var1.append((W[c]**2)*(sigma[c]**2)) 
        except TypeError:
            print "W", W, len(W)
            print "Sigma", sigma, len(sigma)
    for x in itertools.combinations(range(n),2):
        x = list(x)
        try:
            var2.append((2 * W[x[0]] * W[x[1]] * Corr[x[0]][x[1]] * sigma[x[0]] * sigma[x[1]])) 
        except TypeError:
            print "x", x
            print "W", W
            print "Sigma", sigma
            print ""
            print "Corr", Corr,
            print ""
    tvar = sum(var1) + sum(var2)
    return tvar
 
def port_mean_var(n, W, R, C, Corr, sigma):
    return port_mean(W, R), port_var(n, W, Corr, sigma)
 
 
def solve_frontier(R, C, rf, Corr, sigma):
    def fitness(W, R, C, r, Corr, sigma):
        # For given level of return r, find weights which minimizes
        # portfolio variance.
        mean, var = port_mean_var(n, W, R, C, Corr, sigma)
        # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
        penalty = 50*abs(mean-r)
        return var + penalty
    frontier_mean, frontier_var, frontier_weights = [], [], []
    n = len(R)      # Number of assets in the portfolio
    for r in linspace(min(R), max(R), num=20): # Iterate through the range of returns on Y axis
        W = numpy.ones([n])/n         # start optimization with equal weights
        b_ = [(0,1) for i in range(n)]
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })
        # print ""
        # print("n", n)
        # print("c_", c_)
        # print("b_", b_)
        # print("W", W)
        # print("R", R)
        # print("C", C)
        # print("rf", rf)
        optimized = scipy.optimize.minimize(fitness, W, (R, C, r, Corr, sigma), method='SLSQP', constraints=c_, bounds=b_)   
        if not optimized.success: 
            raise BaseException(optimized.message)
        # add point to the min-var frontier [x,y] = [optimized.x, r]
        frontier_mean.append(r)                                                 # return
        frontier_var.append(port_var(n, optimized.x, Corr, sigma))   # min-variance based on optimized weights
        frontier_weights.append(optimized.x)
    return numpy.array(frontier_mean), numpy.array(frontier_var), frontier_weights
 
def solve_weights(R, C, rf, Corr, sigma, n):
    def fitness(W, R, C, rf):
        try:
            mean, var = port_mean_var(n, W, R, C, Corr, sigma)      # calculate mean/variance of the portfolio
            util = (mean - rf) / numpy.sqrt(var)  
        except TypeError:
            print(type(mean))
            print(type(rf))
            print(type(var))
            print(mean)
            print(rf)
            print(var)
            return None
        return 1/util                                           # maximize the utility, minimize its inverse value
    n = len(R)
    W = numpy.ones([n])/n                                         # start optimization with equal weights
    b_ = [(0.,1.) for i in range(n)]        # weights for boundaries between 0%..100%. No leverage, no shorting
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })       # Sum of weights must be 100%
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)  
    if not optimized.success: 
        return None
    else:
        return optimized.x   
     
 
def optimize_and_display(title, names, R, C, rf, Corr, sigma, color='black'):
    n = len(names)
    W = solve_weights(R, C, rf, Corr, sigma, n)
    if W is None:
        W = numpy.ones([n])/n 
    mean, var = port_mean_var(n, W, R, C, Corr, sigma)                        
    sharpe = (mean - rf)/numpy.sqrt(var)
    return mean, var, sharpe, names, W
 
 
 
def get_optimum_portfolios(matrix, portfolio_size, tree_branch_width, tree_root_width, bulk_data, rf, rankings):
    # just to go to next line after the 'update_progess' from portfolio_coordinates
    top_ports = []
    coordinates, coordinates_length = portfolio_coordinates(matrix, portfolio_size, tree_branch_width, tree_root_width, bulk_data, rankings, rf)
    # if coordinates == None and coordinates_length == None:
    #     return None
    count = 0
    for coo in coordinates:
        if len(set(coo[1]))!=portfolio_size:
            count +=1
            coordinates.pop((coordinates.index(coo)))
    if coordinates_length > rankings:
        coordinates = sorted(coordinates, key=lambda x: x[0], reverse=True)
        coordinates = coordinates[:rankings]
    for coordinate in coordinates:
        # print "coordinate", coordinate
        count += 1
        full_coordinate = coordinate
        coordinate = coordinate[1]
        progress = count/float(coordinates_length)
        update_progress(progress, "         Optimizing Portfolios:", "         Portfolios Optimized")
        portfolio = {}
        names, W, R, C, Corr, sigma, exchanges = get_data(bulk_data, coordinate)
        n = len(names)
        if n != portfolio_size:
            print "n != portfolio_size at coordinates", coordinate
            continue
        else:
            # mean,var = port_mean_var(n, W, R, C, Corr, sigma)
            portfolio['mean'], portfolio['var'], portfolio['sharpe'], portfolio['names'], portfolio['weights'] = optimize_and_display('Optimization based on Historical returns', names, R, C, rf, Corr, sigma, color='red')
            portfolio['R'], portfolio['Corr'], portfolio['exchanges'] = R, Corr, exchanges
            portfolio['pos'], portfolio['pos_rank'], portfolio['pos_value'], portfolio['pos_count'] = (coordinate), coordinates.index(full_coordinate), full_coordinate[0], full_coordinate[2]
            if len(portfolio['names']) == len(list(portfolio['names'])):
                top_ports.append(portfolio)
    return sorted(top_ports, key=lambda x: x['sharpe'], reverse=True)
 
 
def portfolio_coordinates(matrix, portfolio_size, tree_branch_width, tree_root_width, WORKING_STOCKS ,RF):
    returnsx = []
    riskx = []
    for stock in bulk_data[:matrix_length]:
        returnsx.append(stock['avg_returns'])
        riskx.append(stock['std_returns'])
    # def nth_smallest_array(a, n):
    #     return numpy.partition(a, n-1)[n-1]
    # def nth_largest_array(a, n):
    #     return -numpy.partition(-a, n)[n]
    returnsx = numpy.array(returnsx)
    returnsy = returnsx[:,None]
    riskx = numpy.array(riskx)
    risky = riskx[:,None]
    weightsm = numpy.zeros(shape=(riskx.size,risky.size))
    weightsm = weightsm + (riskx/(riskx+risky)) # weights are row specific: i.e. row weight is == row/(col+row)
    corrm = matrix
    mult_matrix = ((returnsx*(1-weightsm))+(returnsy*(weightsm))-rf)/numpy.sqrt(((risky**2)*(weightsm**2)) + ((riskx**2)*((1-weightsm)**2)) + (2*riskx*risky*corrm*weightsm*(1-weightsm))) # this creates the correlation matrix multiplied by the summation of their respective individual sharpes
    # We want the smallest values, not the largest values in optimizing portfolios
    mult_matrix = mult_matrix.round(6)
    flat_mult_matrix = numpy.squeeze(numpy.asarray(mult_matrix[~numpy.eye(mult_matrix.shape[0], dtype='bool')]))
    curr_count = 0
    coordinates = []
    n_permutations = (factorial(portfolio_size-2)) * tree_root_width
    max_start_vals = numpy.argpartition(-flat_mult_matrix,tree_root_width) 
    for start in range(tree_root_width):
        max_start_val = flat_mult_matrix[max_start_vals[start]]
        # print ""
        # print "mult_matrix"
        # for i in mult_matrix:
        #     print i
        # print ""
        # print "mult_matrix max = ", mult_matrix.max()
        # print "max_start_val", max_start_val
        max_start_spot = numpy.argwhere(mult_matrix == max_start_val)[0]
        # print "max_start_spot", max_start_spot
        dummy_spot = max_start_spot
        # print "dummy_spot", dummy_spot
        for loop in range(10):
            # print "loop", loop
            if len(dummy_spot) == 1:
                # print "dummy_spot ==1"
                dummy_spot = dummy_spot[0]
                # print "new dummy_spot", dummy_spot
            else:
                # print "else"
                max_start_spot = dummy_spot
                # print "max_start_spot", max_start_spot
                break
        col_a = mult_matrix[:, int(max_start_spot[0])]
        # print "cola", col_a
        col_b = mult_matrix[:, int(max_start_spot[1])]
        # print "colb", col_b
        for tupl in itertools.permutations([x for x in range(portfolio_size-2)]):
            curr_count += 1
            update_progress(curr_count/float(n_permutations), "         Optimizing Coordinates:", "         Coordinates Optimized")
            coordinate = [max_start_val, [int(max_start_spot[0]),int(max_start_spot[1])]]
            if portfolio_size > 2:
                summed = col_a + col_b
                for index in tupl:
                    max_spot = numpy.argpartition(-summed,index)[index]
                    if max_spot in coordinate[1]:
                        for x in range(matrix_length-2):
                            x = x + index + 3
                            max_spot = numpy.argpartition(-summed,index+x)[index+x]
                            if max_spot in coordinate[1]:
                                continue
                            else:
                                break
                    coordinate[1].append(max_spot)
                    coordinate[0] = summed[max_spot] + coordinate[0]
                    summed = summed + mult_matrix[:, int(max_spot)]
            coordinates.append(coordinate)
            coordinate.append(curr_count)
    coo_val = []
    coordinates_cleansed = []
    # print ""
    # print "Pre Coordinates"
    # for i in coordinates:
    #     print i
    # print ""
    for coord in coordinates:
        if set(coord[1]) not in coo_val:
            coordinates_cleansed.append(coord)
            coo_val.append(set(coord[1]))
    coordinates = sorted(coordinates_cleansed, key=lambda x: x[0], reverse=True)
    coordinates_length = len(coordinates)
    # print "Length of coordinates returned: ", coordinates_length
    # for i in coordinates:
    #     print i
    return coordinates, coordinates_length