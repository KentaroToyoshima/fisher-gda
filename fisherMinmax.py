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

def gda_linear(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, update_num, arch, decay_outer = False, decay_inner = False):
    demands = np.copy(demands_0)
    prices = np.copy(prices_0)
    demands_hist = []
    prices_hist = []
    demands_hist.append(np.copy(demands))
    prices_hist.append(np.copy(prices))

    for iter in range(1, num_iters):
        if (not iter % 1000):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        ### Price Step ###
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        if (decay_outer) :
            step_size = learning_rate[0]*(iter**(-1/2))*excess_demand
            prices += step_size*(prices > 0)
        else:
            if arch == 'm-alg2':
                step_size = learning_rate[0]*(excess_demand + mutation_rate*(prices_ref - prices))
            else:
                step_size = learning_rate[0]*excess_demand
            prices += step_size*(prices > 0)
        if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
            prices_ref = np.copy(prices)
        
        prices = prices.clip(min=0.0001)
        prices_hist.append(np.copy(prices))

        ### Demand Step ###
        if (decay_inner):
            demands += learning_rate[1]*iter**(-1/2)*valuations
        else:
            if arch == 'm-alg2':
                demands += learning_rate[1]*(valuations - prices + mutation_rate*(demands_ref - demands))
            elif arch == 'alg2':
                demands += learning_rate[1]*(valuations - prices)
            elif arch == 'alg4':
                demands += learning_rate[1]*valuations
            else:
                print('error')
                exit()
        if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
            demands_ref = np.copy(demands)
        # Projection step
        if arch == 'alg4':
            demands = project_to_bugdet_set(demands, prices, budgets)
        
        demands = demands.clip(min = 0) #Should remove logically but afraid things might break
        demands_hist.append(np.copy(demands))
        
    return (demands, prices, demands_hist, prices_hist)

############### Cobb-Douglas ###############

def gda_cd(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, update_num, arch, decay_outer = False, decay_inner = False):
    demands = np.copy(demands_0).clip(min = 0.001)
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    prices_hist.append(np.copy(prices))
    demands_hist.append(np.copy(demands))

    for iter in range(1, num_iters):
        if (not iter % 1000):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")

        ### Prices Step ###
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1

        if (decay_outer) :
            step_size = learning_rate[0]*(iter**(-1/2))*excess_demand
            prices += step_size*((prices) > 0)
        else:
            if arch == 'm-alg2':
                step_size = learning_rate[0]*(excess_demand + mutation_rate*(prices_ref - prices))
            else:
                step_size = learning_rate[0]*excess_demand
            prices += step_size*((prices) > 0)

        if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
            prices_ref = np.copy(prices)

        prices = prices.clip(min=0.0001)
        prices_hist.append(np.copy(prices))

        ### Demands Step ###
        if (decay_inner):
            demands += learning_rate[1]*iter**(-1/2)*(np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T
        else:
            if arch == 'm-alg2':
                demands += learning_rate[1]*(((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T) - prices + mutation_rate*(demands_ref - demands))
            elif arch == 'alg2':
                demands += learning_rate[1]*(((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T) - prices)
            elif arch == 'alg4':
                demands += learning_rate[1]*((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T)
            else:
                print('error')
                exit()

        if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
            demands_ref = np.copy(demands)

        # Projection step
        if arch == 'alg4':
            demands = project_to_bugdet_set(demands, prices, budgets)

        demands = demands.clip(min = 0)
        demands_hist.append(np.copy(demands))
        

    return (demands, prices, demands_hist, prices_hist)

############# Leontief ###############
 
def gda_leontief(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, update_num, arch, decay_outer = False, decay_inner = False):
    demands = np.copy(demands_0)
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    prices_hist.append(np.copy(prices))
    demands_hist.append(np.copy(demands))
    
    for iter in range(1, num_iters):
        if (not iter % 1000):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        ### Prices Step ###
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        if (decay_outer):
            step_size = learning_rate[0]*iter**(-1/2)*excess_demand
            prices += step_size*((prices) > 0)
        else:
            if arch == 'm-alg2':
                step_size = learning_rate[0]*(excess_demand + mutation_rate*(prices_ref - prices))
            else:
                step_size = learning_rate[0]*excess_demand
            prices += step_size*((prices) > 0)
        if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
            prices_ref = np.copy(prices)

        prices = prices.clip(min=0.00001)
        prices_hist.append(np.copy(prices))

        ### Demands Step ###
        for buyer in range(budgets.shape[0]):
            # Find a good that provides "minimum utility"
            min_util_good = np.argmin(demands[buyer,:]/valuations[buyer,:])
            
            # Gradient Step
            if(decay_inner):
                demands[buyer,min_util_good] += learning_rate[1]*iter**(-1/2)*(1/(valuations[buyer, min_util_good]))
            else:
                if arch == 'm-alg2':
                    demands[buyer,min_util_good] += learning_rate[1]*((1/(valuations[buyer, min_util_good])) - prices[min_util_good] + mutation_rate*(demands_ref[buyer, min_util_good] - demands[buyer, min_util_good]))
                elif arch == 'alg2':
                    demands[buyer,min_util_good] += learning_rate[1]*((1/(valuations[buyer, min_util_good])) - prices[min_util_good])
                elif arch == 'alg4':
                    demands[buyer,min_util_good] += learning_rate[1]*(1/(valuations[buyer, min_util_good]))
                else:
                    print('error')
                    exit()
            if arch == 'm-alg2' and update_num != 0 and iter % update_num == 0:
                demands_ref = np.copy(demands)

        # Projection step
        if arch == 'alg4':
            demands = project_to_bugdet_set(demands, prices, budgets)
        
        demands = demands.clip(min = 0)
        demands_hist.append(np.copy(demands))


    return (demands, prices, demands_hist, prices_hist)

def calc_gda(num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_num, arch, market_type, decay_outer=False, decay_inner=False):
    allocations = np.copy(allocations_0)
    prices = np.copy(prices_0)
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

        step_size = excess_allocation
        if decay_outer:
            step_size *= iter ** (-1 / 2)
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
                #allocations_grad = np.zeros_like(allocations)
                for buyer in range(budgets.shape[0]):
                    min_util_good = np.argmin(allocations[buyer, :] / valuations[buyer, :])
                    allocations_grad[buyer, :] = budgets[buyer]/max(allocations[buyer, min_util_good], 0.001) - prices[min_util_good]
                    #allocations_grad[buyer, min_util_good] = budgets[buyer]/max(allocations[buyer, min_util_good], 0.001) - prices[min_util_good]
                    #allocations_grad[buyer, :] = 1 / valuations[buyer, min_util_good]
                    #allocations_grad[buyer, :] = budgets[buyer]/allocations[buyer, min_util_good] - prices[min_util_good]
                    #allocations_grad[:, min_util_good] = 1 / valuations[buyer, min_util_good]
                    #allocations_grad[buyer, min_util_good] = 1 / valuations[buyer, min_util_good]
            elif arch == 'alg4':
                #allocations_grad = np.zeros_like(allocations)
                for buyer in range(budgets.shape[0]):
                    min_util_good = np.argmin(allocations[buyer, :] / valuations[buyer, :])
                    allocations_grad[buyer, :] = budgets[buyer]/max(allocations[buyer, min_util_good], 0.001)
                    #allocations_grad[buyer, min_util_good] = budgets[buyer]/max(allocations[buyer, min_util_good], 0.001)
                    #allocations_grad[buyer, :] = 1 / valuations[buyer, min_util_good]
                    #allocations_grad[:, min_util_good] = 1 / valuations[buyer, min_util_good]
                    #allocations_grad[buyer, min_util_good] = 1 / valuations[buyer, min_util_good]
            elif arch == 'm-alg2':
                #allocations_grad = np.zeros_like(allocations)
                for buyer in range(budgets.shape[0]):
                    min_util_good = np.argmin(allocations[buyer, :] / valuations[buyer, :])
                    allocations_grad[buyer, :] = budgets[buyer]/max(allocations[buyer, min_util_good], 0.001) - prices[min_util_good]
                    #allocations_grad[buyer, min_util_good] = budgets[buyer]/max(allocations[buyer, min_util_good], 0.001) - prices[min_util_good]
                    #allocations_grad[buyer, :] = 1 / valuations[buyer, min_util_good]
                    #allocations_grad[:, min_util_good] = 1 / valuations[buyer, min_util_good]
                    #allocations_grad[buyer, min_util_good] = 1 / valuations[buyer, min_util_good]
                allocations_grad += mutation_rate * (allocations_ref - allocations)
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