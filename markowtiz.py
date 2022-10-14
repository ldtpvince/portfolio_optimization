import scipy.optimize as optim
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 1000

with open('stocks.txt', 'r') as f:
    stocks_name = f.read().splitlines()

start_date = '2015-01-01'
end_date = '2021-01-01'

# Download data from yahoo finance
def download_data():
    stock_data = None
    treasury_data = None

    if os.path.exists('stock_prices.csv'):
        stock_data = pd.read_csv('stock_prices.csv', index_col='Date')
    else:
        stock_data = {}

        for stock in stocks_name:
            ticker = yf.Ticker(stock)
            stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

        stock_data = pd.DataFrame(stock_data)
        stock_data.to_csv('stock_prices.csv')

    if os.path.exists('treasury.csv'):
        treasury_data = pd.read_csv('treasury.csv', index_col='Date')
    else:
        treasury_data = yf.Ticker('^TNX').history(start=start_date, end=end_date)['Close']
        treasury_data = pd.DataFrame(treasury_data)
        treasury_data.to_csv('treasury.csv')

    return stock_data, treasury_data

# Plot the close prices of the given stocks
def plot_data(data):
    data.plot(figsize=(10, 5))
    plt.show()

# Plot the efficient frontier for the simulated portfolios
def plot_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel("Expected volatilities")
    plt.ylabel("Expected Returns")
    plt.colorbar(label="Sharpe ratio")
    plt.show()

def plot_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel("Expected volatilities")
    plt.ylabel("Expected Returns")
    plt.colorbar(label="Sharpe ratio")
    plt.show()

# Calculate the normalize returns
def cal_returns(data):
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

# Calculate the mean and the variance of the returns annually
def cal_statistic(returns, w):
    annual_returns = np.sum(returns.mean() * w) * NUM_TRADING_DAYS
    annual_volatility = np.sqrt(w.T @ (returns.cov() * NUM_TRADING_DAYS) @ w) 

    return np.array(annual_returns), np.array(annual_volatility)

# The target for sharpe ratio compare to treasury yield of the US
def cal_sharpe_ratio(w, returns, treasury_yield):
    portfolio_return, portfolio_volatilit
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    y = cal_statistic(re
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    turns, w)
    s
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    harpe_ratio = (
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    portfolio_retur
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    n - treasury_yield) / portfolio_volatility
    return sharpe_ratio

def simulate_portfolios(log_returns):
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_weights = []

    for i in range(NUM_PORTFOLIOS):
        w = np.random.rand(len(stocks_name))
        w = w / np.sum(w)
        returns, volatilities = cal_statistic(log_returns, w)
        portfolio_returns.append(returns)
        portfolio_volatilities.append(volatilities)
        portfolio_weights.append(w)

    return np.array(portfolio_returns), np.array(portfolio_volatilities), np.array(portfolio_weights)

# Find the optimal stocks weight to maximize the sharpe ratio or minimize the negative function
def target_function(w, returns, treasury_yield):
    return -cal_sharpe_ratio(w, returns, treasury_yield)

def optimize_portfolios(w, returns, treasury_yield):
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0,1) for _ in range(len(stocks_name)))
    return optim.minimize(fun=target_function, x0=w[0], args=(returns, treasury_yield)
                        , method='SLSQP', bounds=bounds, constraints=constraints)

def print_optimal_portfolios(optimum, returns, treasury_yield):
    p_w = optimum['x'].round(3)
    p_returns, p_volatilities = cal_statistic(returns, p_w)
    print('Optimal portfolio: ', p_w)
    print('Expected return: ', p_returns)
    print('Expected volatilities: ', p_volatilities)
    print('Expected sharpe ratio: ', cal_sharpe_ratio(p_w, returns, treasury_yield)['Close'])

if __name__ == "__main__":
    stocks_data, treasury_data = download_data()
    # plot_data(stocks_data)
    # plot_data(treasury_data)

    log_returns = cal_returns(stocks_data)
    treasury_returns = cal_returns(treasury_data)
    treasury_yield_returns = treasury_returns.mean()

    p_returns, p_volatilities, p_weights = simulate_portfolios(log_returns)
    # plot_portfolios(p_returns, p_volatilities)
    optimum = optimize_portfolios(p_weights, log_returns, treasury_yield_returns)
    print('Treasury yield: ', treasury_yield_returns['Close'])
    print_optimal_portfolios(optimum, log_returns, treasury_yield_returns)
