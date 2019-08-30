import numpy as np

from scipy.optimize import minimize

from models.financial_instruments import WeightedPortfolio


class PortfolioOptimizer(object):
    """
        This class is designed to optimize the weighting of a given portfolio
        in order to maximize the sharpe ratio
    """

    def optimize_portfolio(self, portfolio, rf: float, portfolio_size: int):
        """
            Returns the portfolio with the optimized weights
        """
        
        weights = self._solve_weights(portfolio, rf, portfolio_size)

        if weights is None:
            weights = np.zerosones([portfolio_size])/portfolio_size

        optimized_portfolio = WeightedPortfolio(
            portfolio=portfolio,
            weights=weights,
            risk_free=rf)

        optimized_portfolio.build_weighted_returns_data_series()

        return optimized_portfolio
    

    def _solve_weights(self, portfolio, rf, portfolio_size):
        def fitness(weights, portfolio, rf):
            weighted_portfolio = WeightedPortfolio(
                portfolio=portfolio,
                weights=weights,
                risk_free=rf)
            utility = weighted_portfolio.portfolio_sharpe
            return 1/utility

        # start optimization with equal weights
        weights = np.ones([portfolio_size])/portfolio_size
        # weights must be between 0 - 100%, assuming no shorting or leveraging
        bounds = [(0.,1.) for i in range(portfolio_size)]
        # Sum of weights must be 100%
        constraints = ({'type':'eq', 'fun': lambda weights: sum(weights) - 1.0})       
        optimized = minimize(fitness, weights, (portfolio, rf), method='SLSQP', constraints=constraints, bounds=bounds)
        if not optimized.success:
            return None
        else:
            return optimized.x