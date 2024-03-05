# Library Imports
import numpy as np
import cvxpy as cp
np.set_printoptions(precision=3)

def get_linear_utility(allocation, valuations):
    return allocation.T @ valuations

def get_CD_utility(allocation, valuations):
    return np.prod(np.power(allocation, valuations))

def get_leontief_utility(allocation, valuations):
    return np.min(allocation/valuations)

def get_linear_indirect_util(prices, budget, valuations):
    return np.max(valuations/prices)*budget

def get_CD_marshallian_demand(prices, budget, valuations):
    normalized_vals = valuations / np.sum(valuations)
    return normalized_vals*budget/prices.clip(min = 0.00001)

def get_CD_indirect_util(prices, budget, valuations):
    return get_CD_utility(get_CD_marshallian_demand(prices, budget, valuations), valuations)

def get_leontief_indirect_util(prices, budget, valuations):
    return budget/(prices.T @ valuations)
