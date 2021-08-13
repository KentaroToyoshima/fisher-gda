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

def gda_linear(valuations, budgets, prices_0, learning_rate , num_iters, decay_outer = False, decay_inner = False):
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    demands = np.zeros(valuations.shape)  
    for iter in range(1, num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        
        prices_hist.append(prices)
        
        ### Demand Step ###    
        # Gradient Step
        if (decay_outer):
            demands += iter**(-1/2)*valuations
        else:    
            demands += valuations
        
        # Projection step
        demands = project_to_bugdet_set(demands, prices, budgets)

        
        demands = demands.clip(min = 0) #Should remove logically but afraid things might break
        demands_hist.append(demands)
        
        ### Price Step ###
        
        # Gradient Step
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        if (decay_outer) :
            step_size = learning_rate*(iter**(-1/2))*excess_demand
            prices += step_size*(prices > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*(prices > 0)
        
        prices = prices.clip(min=0.01)
        

    return (demands, prices, demands_hist, prices_hist)

############### Cobb-Douglas ###############

def gda_cd(valuations, budgets, prices_0, learning_rate, num_iters, decay_outer = False, decay_inner = False):
    prices = np.copy(prices_0)
    prices_hist = []
    demands_hist = []
    demands = np.zeros(valuations.shape).clip(min = 0.001)  

    for iter in range(1, num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        prices_hist.append(prices)
        
        ### Demands Step ###

        # Gradient Step
        if (decay_inner):
            demands += iter**(-1/2)*(np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T
        else:
            demands += (np.prod(np.power(demands, valuations), axis = 1)*(valuations/demands.clip(min = 0.001)).T).T
        
        # Projection step
        demands = project_to_bugdet_set(demands, prices, budgets)

        demands = demands.clip(min = 0)        
        demands_hist.append(demands)
        
        
        ### Prices Step ###
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1

        if (decay_outer) :
            step_size = learning_rate*(iter**(-1/2))*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*((prices) > 0)

        prices = prices.clip(min=0.01)
        

    return (demands, prices, demands_hist, prices_hist)

############# Leontief ###############
 
def gda_leontief(valuations, budgets, prices_0, learning_rate, num_iters, decay_outer = False, decay_inner = False):
    prices = prices_0
    prices_hist = []
    demands_hist = []
    
    demands = np.zeros(valuations.shape)
    for iter in range(1, num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")
        
        prices_hist.append(prices)    
    
        ### Demands Step ###
        for buyer in range(budgets.shape[0]):
            # Find a good that provides "minimum utility"
            min_util_good = np.argmin(demands[buyer,:]/valuations[buyer,:])
            
            # Gradient Step
            if(decay_inner):
                demands[buyer,min_util_good] += iter**(-1/2)*(1/(valuations[buyer, min_util_good]))
            else:  
                demands[buyer,min_util_good] += (1/(valuations[buyer, min_util_good]))

        # Projection step
        demands = project_to_bugdet_set(demands, prices, budgets)
        
     
        demands = demands.clip(min = 0)
        demands_hist.append(demands)
        
        ### Prices Step ###
        
        demand = np.sum(demands, axis = 0)
        excess_demand = demand - 1
        
        if (decay_outer) :
            step_size = learning_rate*iter**(-1/2)*excess_demand
            prices += step_size*((prices) > 0)
        else:
            step_size = learning_rate*excess_demand
            prices += step_size*((prices) > 0)

        prices = prices.clip(min=0.00001)
        
        
    return (demands, prices, demands_hist, prices_hist)
