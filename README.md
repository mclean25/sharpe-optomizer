# Sharpe Optimizer

This program attempts to find the portfolios with the highest sharpe ratios for a given time span an stock universe. When these portfolios are returned, you can evaluate their "post-bought" performance by viewing their `future_data_frame` attribute.

### Getting Started

Suppors Python 3.x

1. Install the required packages: `pip install -r requirements.txt`.
2. Execute the program: `python main.py`.

If this runs without errors, your system should be good to go.

#### Viewing the Output

I havn't thought of a good way to output the results yet that is scalable and easy to injest into another Python program or Excel to perform analysis on the results. Therfore for now I suggest using [Visual Studio Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/) to run the program, and then set a breakpoint right before the program exists to view the `optimized_portfolios` in memory.

If you have suggestions for how the output should look, feel free to [write up an issue](https://github.com/mclean25/sharpe-optomizer/issues).

### Stock Universe Input

By default, `symbols_short_list.csv` is used to provide a list of tickers for the stock universe. This list is purposefully small so you can execute the full program quickly to spot any errors. However, you can use any of the other `csv` files in `./examples/`. To input your own `.csv` file of tickers into the program, run `python main.py "C:\\path\\to\\my\\csv\\file.csv"`.

#### Data Source & Caching

Yahoo finance is currently used as the data source for pulling stock price data on the provided tickers. If you provide a large list of tickers to download, it will take the program some time to download all of the pricing data. To overcome this when running the program multiple times over the same tickers, the pricing data is cached into `./cached_stock_data.sqlite`. As long as the data in the database isn't more than a few days old, the program will pull the pricing data from there if it exists.

### How it Works
1. Stock universe is loaded in
2. Stocks with an independant sharpe `< 0` and that do not exist for the full historical time span are filtered out.
3. Since the program quickly becomes an NP-Hard (eg: travelling salesman) problem once the number of stocks in the universe reaches a significant amount, we implement the `universe_optimizer.py` which could be considered the "secret sauce" of the program. To summarize, this part of the program will build `portfolio_candidates` based on a 2-stock sharpe ratio matrix of the stock universe.
4. Once we have a list of `portfolio_candidates`, we use linear optimization to find the weightings for each portfolio which maximize that portfolios sharpe ratio.
5. The ranked portoflios of highest to lowest sharpe ratios is then returned.