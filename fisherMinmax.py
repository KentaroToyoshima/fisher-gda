# Library Imports
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import consumerUtility as cu
np.set_printoptions(precision=3)

#TODO:将来的に，確率的勾配降下法の実装も考えたほうが良い？
############# Projection onto affine positive half-space, i.e., budget set ###############
def project_to_bugdet_set(X, p, b):
    X_prec = X
    while (True): 
        X -= ((X @ p - b).clip(min= 0)/(np.linalg.norm(p)**2).clip(min= 0.01) * np.tile(p, reps = (b.shape[0], 1)).T).T
        X = X.clip(min = 0)
        if(np.linalg.norm(X - X_prec) <= np.sum(X_prec)*0.05):
            break
        # print(f"Current iterate {X}\nPrevious Iterate {X_prec}")
        X_prec = X
    return X
############# Linear ###############

def calc_gda(num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_num, arch, market_type, decay_outer=False, decay_inner=False):
    prices = np.copy(prices_0)
    allocations = np.copy(allocations_0)
    prices_hist = []
    allocations_hist = []
    prices_hist.append(np.copy(prices))
    allocations_hist.append(np.copy(allocations))

    for iter in range(1, num_iters):
        if not iter % 1000:
            print(f" ----- Iteration {iter}/{num_iters} ----- ")

        # Price Step
        allocation = np.sum(allocations, axis=0)
        excess_allocation = allocation - 1

        step_size = np.copy(excess_allocation)
        # if decay_outer:
        #     step_size *= iter ** (-1 / 2)
        if arch == 'm-alg2':
            step_size += mutation_rate * (prices_ref - prices)
        prices += learning_rate[0] * step_size * (prices > 0)
        if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
            prices_ref = np.copy(prices)
        prices = prices.clip(min=0.0001)
        prices_hist.append(np.copy(prices))

        # allocations Step
        allocations_grad = np.zeros_like(allocations)
        if market_type == 'linear':
            if arch == 'alg2':
                allocations_grad = (budgets/(np.sum(valuations*allocations, axis = 1))*valuations.T).T - np.array([prices, ]*budgets.shape[0])
            elif arch == 'alg4':
                #allocations_grad = np.copy(valuations)
                allocations_grad = (budgets/(np.sum(valuations*allocations, axis = 1))*valuations.T).T
            elif arch == 'm-alg2':
                #allocations_grad = np.copy(valuations) - prices + mutation_rate * (allocations_ref - allocations)
                allocations_grad = (budgets/(np.sum(valuations*allocations, axis = 1))*valuations.T).T - np.array([prices, ]*budgets.shape[0]) + mutation_rate * (allocations_ref - allocations)
        elif market_type == 'cd':
            if arch == 'alg2':
                allocations_grad = (budgets*(valuations/allocations.clip(min=0.001)).T).T - np.array([prices, ]*budgets.shape[0])
            elif arch == 'alg4':
                #allocations_grad = (np.prod(np.power(allocations, valuations), axis=1) * (valuations / allocations.clip(min=0.0001)).T).T
                allocations_grad = (budgets*(valuations/allocations.clip(min=0.001)).T).T
            elif arch == 'm-alg2':
                #allocations_grad = (np.prod(np.power(allocations, valuations), axis=1) * (valuations / allocations.clip(min=0.0001)).T).T - prices + mutation_rate * (allocations_ref - allocations)
                allocations_grad = (budgets*(valuations/allocations.clip(min=0.001)).T).T - np.array([prices, ]*budgets.shape[0]) + mutation_rate * (allocations_ref - allocations)
        elif market_type == 'leontief':
            if arch == 'alg2':
                for buyer in range(budgets.shape[0]):
                    min_util_good = np.argmin(allocations[buyer, :] / valuations[buyer, :])
                    allocations_grad[buyer, min_util_good] = budgets[buyer]/max(allocations[buyer, min_util_good], 1e-5) - prices[min_util_good]
            elif arch == 'alg4':
                for buyer in range(budgets.shape[0]):
                    min_util_good = np.argmin(allocations[buyer, :] / valuations[buyer, :])
                    allocations_grad[buyer, min_util_good] = budgets[buyer]/max(allocations[buyer, min_util_good], 1e-5)
            elif arch == 'm-alg2':
                for buyer in range(budgets.shape[0]):
                    min_util_good = np.argmin(allocations[buyer, :] / valuations[buyer, :])
                    allocations_grad[buyer, min_util_good] = budgets[buyer]/max(allocations[buyer, min_util_good], 1e-5) - prices[min_util_good]
                    allocations_grad[buyer, :] += mutation_rate * (allocations_ref[buyer, :] - allocations[buyer, :])
        else: exit("unknown market type")

        # if decay_inner:
        #     allocations_grad *= iter ** (-1 / 2)
        # if arch == 'm-alg2':
        #     allocations_grad += mutation_rate * (allocations_ref - allocations)
        # if arch != 'alg4':
        #     allocations_grad -= prices

        allocations += learning_rate[1] * allocations_grad
        if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
            allocations_ref = np.copy(allocations)

        # Projection step
        if arch == 'alg4':
            allocations = project_to_bugdet_set(allocations, prices, budgets)
        allocations = allocations.clip(min=0)
        allocations_hist.append(np.copy(allocations))

    return (allocations, prices, allocations_hist, prices_hist)