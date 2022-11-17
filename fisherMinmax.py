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

def gda_linear(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, decay_outer = False, decay_inner = False):
    demands = np.copy(demands_0)
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    for iter in range(1, num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
       
        
        ### Demand Step ###    
        # Gradient Step
        if (decay_inner):
            demands += learning_rate[1]*iter**(-1/2)*valuations
        else:
            demands += learning_rate[1]*(valuations - num_buyers*prices + mutation_rate*(demands_ref - demands))
            #demands += learning_rate[1]*(valuations - num_buyers*prices)
            #demands += learning_rate[1]*valuations
        
        # Projection step
        # demands = project_to_bugdet_set(demands, prices, budgets)

        
        demands = demands.clip(min = 0) #Should remove logically but afraid things might break
        demands_hist.append(demands)
        
        ### Price Step ###
        # Gradient Step
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        if (decay_outer) :
            step_size = learning_rate[0]*(iter**(-1/2))*excess_demand
            prices += step_size*(prices > 0)
        else:
            step_size = learning_rate[0]*(excess_demand + mutation_rate*(prices_ref - prices))
            #step_size = learning_rate[0]*excess_demand
            prices += step_size*(prices > 0)
        
        prices = prices.clip(min=0.0001)
        prices_hist.append(prices)
        

    return (demands, prices, demands_hist, prices_hist)

############### Cobb-Douglas ###############

def gda_cd(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, decay_outer = False, decay_inner = False):
    demands = np.copy(demands_0).clip(min = 0.001)
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []

    for iter in range(1, num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        ### Demands Step ###

        # Gradient Step
        if (decay_inner):
            demands += learning_rate[1]*iter**(-1/2)*(np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T
        else:
            demands += learning_rate[1]*(((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T) - num_buyers*prices + mutation_rate*(demands_ref - demands))
            #demands += learning_rate[1]*(((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T)-num_buyers*prices)
            #demands += learning_rate[1]*((np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T)
        
        # Projection step
        # demands = project_to_bugdet_set(demands, prices, budgets)

        demands = demands.clip(min = 0)
        demands_hist.append(demands)
        
        
        ### Prices Step ###
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1

        if (decay_outer) :
            step_size = learning_rate[0]*(iter**(-1/2))*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate[0]*(excess_demand + mutation_rate*(prices_ref - prices))
            #step_size = learning_rate[0]*excess_demand
            prices += step_size*((prices) > 0)


        prices = prices.clip(min=0.0001)
        prices_hist.append(prices)

    return (demands, prices, demands_hist, prices_hist)

############# Leontief ###############
 
def gda_leontief(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, decay_outer = False, decay_inner = False):
    demands = np.copy(demands_0)
    prices = prices_0
    prices_hist = []
    demands_hist = []
    
    for iter in range(1, num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        ### Demands Step ###
        for buyer in range(budgets.shape[0]):
            # Find a good that provides "minimum utility"
            min_util_good = np.argmin(demands[buyer,:]/valuations[buyer,:])
            
            # Gradient Step
            if(decay_inner):
                demands[buyer,min_util_good] += learning_rate[1]*iter**(-1/2)*(1/(valuations[buyer, min_util_good]))
            else:  
                demands[buyer,min_util_good] += learning_rate[1]*((1/(valuations[buyer, min_util_good]))-num_buyers*prices[min_util_good] + mutation_rate*(demands_ref[buyer, min_util_good] - demands[buyer, min_util_good]))
                #demands[buyer,min_util_good] += learning_rate[1]*((1/(valuations[buyer, min_util_good]))-num_buyers*prices[min_util_good])
                #demands[buyer,min_util_good] += learning_rate[1]*(1/(valuations[buyer, min_util_good]))

        # Projection step
        # demands = project_to_bugdet_set(demands, prices, budgets)
        
        demands = demands.clip(min = 0)
        
        demands_hist.append(demands)
        
        ### Prices Step ###
        
        demand = np.sum(demands, axis = 0)
        # print(f"demands {demand}")
        excess_demand = demand - 1
        
        if (decay_outer):
            step_size = learning_rate[0]*iter**(-1/2)*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate[0]*(excess_demand + mutation_rate*(prices_ref - prices))
            #step_size = learning_rate[0]*excess_demand
            prices += step_size*((prices) > 0)
        
        prices = prices.clip(min=0.00001)
        # print(prices)
        prices_hist.append(prices)

    return (demands, prices, demands_hist, prices_hist)


#def vgda_linear(valuations, budgets, prices_0, learning_rate , num_iters, decay_outer = False, decay_inner = False):
#    prices = np.copy(prices_0)
#    prices_hist = []
#    demands_hist = []
#    demands = np.zeros(valuations.shape).clip(min =0.01)
#    for iter in range(1, num_iters):
#        if (not iter % 500):
#            print(f" ----- Iteration {iter}/{num_iters} ----- ")
#        
#        ### Demand Step ###    
#        # Gradient Step
#        if (decay_outer):
#            demands += learning_rate[1]*iter**(-1/2)*((budgets/(np.sum(valuations*demands, axis = 1)).clip(min = 0.001)*valuations.T).T - np.array([prices, ]*budgets.shape[0]))
#        else:    
#            demands += learning_rate[1]*((budgets/(np.sum(valuations*demands, axis = 1))*valuations.T).T - np.array([prices, ]*budgets.shape[0]))
#        
#        # Projection step
#        # demands = project_to_bugdet_set(demands, prices, budgets)
#
#        
#        demands = demands.clip(min = 0)
#        demands_hist.append(demands)
#        
#        ### Price Step ###
#        
#        # Gradient Step
#        demand = np.sum(demands, axis = 0)
#        excess_demand = demand - 1
#        
#        if (decay_outer) :
#            step_size = learning_rate[0]*(iter**(-1/2))*excess_demand
#            prices += step_size
#        else:
#            step_size = learning_rate[0]*excess_demand
#            prices += step_size
#        
#        prices = prices.clip(min=0.0001)
#        prices_hist.append(prices)
#        
#
#    return (demands, prices, demands_hist, prices_hist)
#
#
################ Cobb-Douglas ###############
#
#def vgda_cd(valuations, budgets, prices_0, learning_rate, num_iters, decay_outer = False, decay_inner = False):
#    prices = np.copy(prices_0)
#    prices_hist = []
#    demands_hist = []
#    demands = np.zeros(valuations.shape).clip(min = 0.001)  
#
#    for iter in range(1, num_iters):
#        if (not iter % 500):
#            print(f" ----- Iteration {iter}/{num_iters} ----- ")
#        
#        ### Demands Step ###
#
#        # Gradient Step
#        if (decay_inner):
#            demands += learning_rate[1]*iter**(-1/2)*((budgets*(valuations/demands.clip(min = 0.001)).T).T - np.array([prices, ]*budgets.shape[0]))
#        else:
#            demands += learning_rate[1]*((budgets*(valuations/demands).T).T - np.array([prices, ]*budgets.shape[0]))
#        
#        # Projection step
#        # demands = project_to_bugdet_set(demands, prices, budgets)
#
#        demands = demands.clip(min = 0.01)        
#        demands_hist.append(demands)
#        
#        
#        ### Prices Step ###
#        demand = np.sum(demands, axis = 0)
#        excess_demand = demand - 1
#
#        if (decay_outer) :
#            step_size = learning_rate[0]*(iter**(-1/2))*excess_demand
#            prices += step_size*((prices) > 0)
#        else:
#            step_size = learning_rate[0]*excess_demand
#            prices += step_size
#
#
#        prices = prices.clip(min=0.0001)
#        prices_hist.append(prices)
#
#    return (demands, prices, demands_hist, prices_hist)
#
############## Leontief ###############
# 
#def vgda_leontief(valuations, budgets, prices_0, learning_rate, num_iters, decay_outer = False, decay_inner = False):
#    prices = prices_0
#    prices_hist = []
#    demands_hist = []
#    
#    demands = np.zeros(valuations.shape)
#    for iter in range(1, num_iters):
#        if (not iter % 500):
#            print(f" ----- Iteration {iter}/{num_iters} ----- ")
#        
#          
#    
#        ### Demands Step ###
#        for buyer in range(budgets.shape[0]):
#            # Find a good that provides "minimum utility"
#            min_util_good = np.argmin(demands[buyer,:]/valuations[buyer,:])
#            
#            # Gradient Step
#            if(decay_inner):
#                demands[buyer,min_util_good] += learning_rate[1]*iter**(-1/2)*((budgets[buyer]*demands[buyer, min_util_good]))
#                demands[buyer,:] -= prices
#            else:  
#                demands[buyer,min_util_good] += learning_rate[1]**((budgets[buyer]*demands[buyer, min_util_good]) - prices[min_util_good])
#
#        # Projection step
#        # demands = project_to_bugdet_set(demands, prices, budgets)
#        
#     
#        demands = demands.clip(min = 0)
#        
#        demands_hist.append(demands)
#        
#        ### Prices Step ###
#        
#        demand = np.sum(demands, axis = 0)
#        # print(f"demands {demand}")
#        excess_demand = demand - 1
#        
#        if (decay_outer) :
#            step_size = learning_rate[0]*iter**(-1/2)*excess_demand
#            prices += step_size*((prices) > 0)
#        else:
#            step_size = learning_rate[0]*excess_demand
#            prices += step_size*((prices) > 0)
#
#        prices = prices.clip(min=0.00001)
#        prices_hist.append(prices)  
#
#    return (demands, prices, demands_hist, prices_hist)
#
#def gdad_linear(valuations, budgets, prices_0, learning_rate , num_iters, decay_outer = False, decay_inner = False):
#    prices = np.copy(prices_0)
#    prices_hist = []
#    demands_hist = []
#    demands = np.zeros(valuations.shape)  
#    bang_per_bucks = np.zeros(valuations.shape[0])
#    
#    for iter in range(1, num_iters):
#        if (not iter % 500):
#            print(f" ----- Iteration {iter}/{num_iters} ----- ")
#        
#        
#        prices_hist.append(prices)
#        
#        ### Bang-per-buck Step ###
#
#        if (decay_outer):
#            bang_per_bucks += iter**(-1/2)*(demands @ prices - budgets)
#        else:
#            bang_per_bucks += demands @ prices - budgets
#        
#        ### Demand Step ###    
#        
#        # Gradient Step
#        if (decay_outer):
#            demands += iter**(-1/2)*(valuations.T/(np.sum(demands*valuations, axis = 1)*budgets)).T
#        else:    
#            demands += (valuations.T/(np.sum(demands*valuations, axis = 1)*budgets)).T
#        
#        # Projection step
#        # demands = project_to_bugdet_set(demands, prices, budgets)
#
#        
#        demands = demands.clip(min = 0) #Should remove logically but afraid things might break
#        demands_hist.append(demands)
#        
#        ### Price Step ###
#        
#        # Gradient Step
#        demand = demands.T @ bang_per_bucks
#        excess_demand = demand - 1
#        
#        
#        if (decay_outer) :
#            step_size = learning_rate*(iter**(-1/2))*excess_demand
#            prices += step_size*(prices > 0)
#        else:
#            step_size = learning_rate*excess_demand
#            prices += step_size*(prices > 0)
#        
#        prices = prices.clip(min=0.00001)
#        
#    return (demands, prices, demands_hist, prices_hist)