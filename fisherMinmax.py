# Library Imports
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import consumerUtility as cu
np.set_printoptions(precision=3)


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
    prices_hist = []
    demands_hist = []
    prices_hist.append(np.copy(prices))
    demands_hist.append(np.copy(demands))

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
                demands += learning_rate[1]*(valuations - num_buyers*prices + mutation_rate*(demands_ref - demands))
            elif arch == 'alg2':
                demands += learning_rate[1]*(valuations - num_buyers*prices)
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
                demands += learning_rate[1]*(((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T) - num_buyers*prices + mutation_rate*(demands_ref - demands))
            elif arch == 'alg2':
                demands += learning_rate[1]*(((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T)-num_buyers*prices)
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
                    demands[buyer,min_util_good] += learning_rate[1]*((1/(valuations[buyer, min_util_good])) - num_buyers*prices[min_util_good] + mutation_rate*(demands_ref[buyer, min_util_good] - demands[buyer, min_util_good]))
                elif arch == 'alg2':
                    demands[buyer,min_util_good] += learning_rate[1]*((1/(valuations[buyer, min_util_good]))-num_buyers*prices[min_util_good])
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