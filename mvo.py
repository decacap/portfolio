import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Define the symbols to download
symbols = ['aapl','goog','tlt','msft','ttt']

# Download the data from Yahoo Finance
data = yf.download(symbols, start='2015-01-01')['Adj Close']
returns = data.pct_change().dropna()

# Define the risk-free rate
risk_free_rate = 0.01

# Define the objective function to minimize
def portfolio_variance(weights, returns):
    return np.dot(weights.T, np.dot(returns.cov(), weights))

# Define the constraint that the weights must sum to 1
def weight_constraint(weights):
    return np.sum(weights) - 1

# Define the initial guess for the weights
n_assets = len(symbols)
weights_0 = np.random.rand(n_assets)
weights_0 /= np.sum(weights_0)

# Define the bounds for the weights
bounds = [(0, 1) for i in range(n_assets)]

# Define the constraints for the weights
constraints = ({'type': 'eq', 'fun': weight_constraint})

# Minimize the objective function
result = minimize(portfolio_variance, weights_0, args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

# Calculate the expected portfolio return
expected_return = np.sum(result.x * returns.mean()) * 252

# Calculate the portfolio volatility
portfolio_volatility = np.sqrt(result.fun) * np.sqrt(252)

# Calculate the Sharpe ratio
sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility

# Print the results
print('Weights: ', result.x)
print('Expected Return: ', expected_return)
print('Portfolio Volatility: ', portfolio_volatility)
print('Sharpe Ratio: ', sharpe_ratio)



