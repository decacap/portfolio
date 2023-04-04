import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Define the symbols for the assets in the portfolio
symbols = ['aapl','goog','tlt','msft','ttt']

# Define the risk-free rate
rf = 0.01

# Download the historical price data for the assets
prices = yf.download(symbols, start='2015-01-01')['Adj Close']

# Calculate the daily returns of the assets
returns = prices.pct_change().dropna()

# Calculate the expected returns and covariances of the assets
mu = returns.mean()
Sigma = returns.cov()
    
    # Define the objective function for the optimizer
def objective(weights):
    port_return = np.dot(weights, mu) * 252 # 252 trading days in a year
    port_var = np.dot(weights, np.dot(Sigma, weights)) * 252 # annualized variance
    port_sharpe = (port_return - rf) / np.sqrt(port_var)
    return -port_sharpe
 

# Define a new function to calculate the Sharpe ratio
def sharpe_ratio(weights):
    port_return = np.dot(weights, mu) * 252 # 252 trading days in a year
    port_var = np.dot(weights, np.dot(Sigma, weights)) * 252 # annualized variance
    port_sharpe = (port_return - rf) / np.sqrt(port_var)
    return port_sharpe

# Define the constraints for the optimizer
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Define the bounds for the optimizer
bounds = [(0, 1) for i in range(len(symbols))]

# Generate random portfolios and find the optimal one
np.random.seed(0)
num_portfolios = 1000
port_returns = []
port_vars = []
port_weights = []
port_sharpes = []

for i in range(num_portfolios):
    weights = np.random.uniform(0, 1, len(symbols))
    weights /= np.sum(weights)
    port_return = np.dot(weights, mu) * 252 # 252 trading days in a year
    port_var = np.dot(weights, np.dot(Sigma, weights)) * 252 # annualized variance
    port_returns.append(port_return)
    port_vars.append(port_var)
    port_weights.append(weights)
    port_sharpe = sharpe_ratio(weights)
    port_sharpes.append(port_sharpe)

init_weights = np.random.uniform(0, 1, len(symbols))
init_weights /= np.sum(init_weights)
result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Output the weights, annualized returns, and Sharpe ratio of the assets in the optimal portfolio
opt_weights = result.x
opt_return = np.dot(opt_weights, mu) * 252 # 252 trading days in a year
opt_var = np.dot(opt_weights, np.dot(Sigma, opt_weights)) * 252 # annualized variance
opt_sharpe = opt_return / np.sqrt(opt_var)
print('Optimal Portfolio Weights:')
for i in range(len(symbols)):
    print(symbols[i] + ': ' + str(round(opt_weights[i]*100, 2)) + '%')
print('Optimal Portfolio Annualized Return: ' + str(round(opt_return, 2)))
print('Optimal Portfolio Sharpe Ratio: ' + str(round(opt_sharpe, 2)))

# Plot the portfolio frontier
import matplotlib.pyplot as plt
plt.scatter(port_vars, port_returns, c=np.array(port_sharpes))
plt.scatter(opt_var, opt_return, marker='*', s=500, c='r')
plt.xlabel('Portfolio Variance')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe Ratio')
plt.show()
