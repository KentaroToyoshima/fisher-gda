
import fisherMinmax as fm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import consumerUtility as cu
from datetime import date

# Objective Functions for linear, Cobb-Douglas, and Leontief

def get_obj_linear(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_linear_indirect_utill(prices.clip(min= 0.0001), b, v)
    # return np.sum(prices) + budgets.T @ np.log(utils) + np.sum(budgets) - np.sum(demands @ prices )
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.01))

def get_obj_cd(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_CD_indirect_util(prices.clip(min= 0.0001), b, v)
    
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001))

def get_obj_leontief(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_leontief_indirect_util(prices.clip(min= 0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001)) 

# Function that run Max-Oracle Gradient Descent and Nested Gradient Descent Ascent Tests and returns data
def run_test(num_buyers, num_goods, learning_rate_linear, learning_rate_cd, learning_rate_leontief, num_experiments, num_iters_linear , num_iters_cd, num_iters_leontief):
    
    prices_hist_gda_linear_all_low = []
    demands_hist_gda_linear_all_low = []
    obj_hist_gda_linear_all_low = []
    prices_hist_gda_cd_all_low = []
    demands_hist_gda_cd_all_low = []
    obj_hist_gda_cd_all_low = []
    prices_hist_gda_leontief_all_low = []
    demands_hist_gda_leontief_all_low = []
    obj_hist_gda_leontief_all_low = []
    prices_hist_gda_linear_all_high = []
    demands_hist_gda_linear_all_high = []
    obj_hist_gda_linear_all_high = []
    prices_hist_gda_cd_all_high = []
    demands_hist_gda_cd_all_high = []
    obj_hist_gda_cd_all_high = []
    prices_hist_gda_leontief_all_high = []
    demands_hist_gda_leontief_all_high = []
    obj_hist_gda_leontief_all_high = []

    for experiment_num in range(num_experiments):

        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")

        # Initialize random market parameters
        valuations = np.random.rand(num_buyers, num_goods)*10 + 5
        budgets = np.random.rand(num_buyers)*10 + 10
        
        
        print(f"****** Market Parameters ******\nval = {valuations}\n budgets = {budgets}\n")
        print(f"*******************************")
        
        print(f"------------ GDA ------------")
        
        print(f"------ Linear Fisher Market ------\n")
        prices_0  = np.random.rand(num_goods)*10 + 5
        #prices_0  = np.full(num_goods, 100, dtype=float)
        print(prices_0)
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_linear(valuations, budgets, prices_0, learning_rate_linear[0], num_iters_linear)
        
        prices_hist_gda_linear_all_low.append(prices_gda)
        demands_hist_gda_linear_all_low.append(demands_gda)
        objective_values = []
        for i in range(0, len(demands_hist_gda)):
            x = np.mean(np.array(demands_hist_gda[:i+1]).clip(min = 0), axis = 0)
            p = np.mean(np.array(prices_hist_gda[:i+1]).clip(min = 0), axis = 0)
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        
        obj_hist_gda_linear_all_low.append(objective_values)
        
        
        print(f"------ Cobb-Douglas Fisher Market ------")
        prices_0  = np.random.rand(num_goods) + 5
        valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T # Normalize valuations for Cobb-Douglas
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_cd(valuations_cd, budgets, prices_0, learning_rate_cd[0], num_iters_cd)
        prices_hist_gda_cd_all_low.append(prices_gda)
        demands_hist_gda_cd_all_low.append(demands_gda)
        objective_values = []
        
        for i in range(0, len(demands_hist_gda)):
            x = np.mean(np.array(demands_hist_gda[:i+1]).clip(min = 0), axis = 0)
            p = np.mean(np.array(prices_hist_gda[:i+1]).clip(min = 0), axis = 0)
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_gda_cd_all_low.append(objective_values)
        
        print(f"------ Leontief Fisher Market ------")
        
        prices_0  = np.random.rand(num_goods) + 10
        
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_leontief(valuations, budgets, prices_0, learning_rate_leontief[0], num_iters_leontief)
        prices_hist_gda_leontief_all_low.append(prices_gda)
        demands_hist_gda_leontief_all_low.append(demands_gda)
        objective_values = []
        for i in range(0, len(demands_hist_gda)):
            x = np.mean(np.array(demands_hist_gda[:i+1]).clip(min = 0), axis = 0)
            p = np.mean(np.array(prices_hist_gda[:i+1]).clip(min = 0), axis = 0)
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_gda_leontief_all_low.append(objective_values)        
        

        # print(f"------------ VGDA ------------")

        # print(f"------ Linear Fisher Market ------\n")
        
        # prices_0  = np.random.rand(num_goods)*10 + 5
        # demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.vgda_linear(valuations, budgets, prices_0, learning_rate_linear[1], num_iters_linear)
        
        # prices_hist_gda_linear_all_high.append(prices_gda)
        # demands_hist_gda_linear_all_high.append(demands_gda)
        # objective_values = []
        # for i in range(0, len(demands_hist_gda)):
        #     x = np.mean(np.array(demands_hist_gda[:i+1]).clip(min = 0), axis = 0)
        #     p = np.mean(np.array(prices_hist_gda[:i+1]).clip(min = 0), axis = 0)
        #     obj = get_obj_linear(p, x, budgets, valuations)
        #     objective_values.append(obj)
        # obj_hist_gda_linear_all_high.append(objective_values)
        
        # print(f"------ Cobb-Douglas Fisher Market ------")
        
        # # Normalize valuations for Cobb-Douglas
        # valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T 
        
        # demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.vgda_cd(valuations_cd, budgets, prices_0, learning_rate_cd[1], num_iters_cd)
        # prices_hist_gda_cd_all_high.append(prices_gda)
        # demands_hist_gda_cd_all_high.append(demands_gda)
        # objective_values = []
        
        # for i in range(0, len(demands_hist_gda)):
        #     x = np.mean(np.array(demands_hist_gda[:i+1]).clip(min = 0), axis = 0)
        #     p = np.mean(np.array(prices_hist_gda[:i+1]).clip(min = 0), axis = 0)
        #     obj = get_obj_cd(p, x, budgets, valuations_cd)
        #     objective_values.append(obj)
        # obj_hist_gda_cd_all_high.append(objective_values)
        
        # print(f"------ Leontief Fisher Market ------")
        
        # demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.vgda_leontief(valuations, budgets, prices_0, learning_rate_leontief[1], num_iters_leontief)
        # prices_hist_gda_leontief_all_high.append(prices_gda)
        # demands_hist_gda_leontief_all_high.append(demands_gda)
        # objective_values = []
        # for i in range(0, len(demands_hist_gda)):
        #     x = np.mean(np.array(demands_hist_gda[:i+1]).clip(min = 0), axis = 0)
        #     p = np.mean(np.array(prices_hist_gda[:i+1]).clip(min = 0), axis = 0)
        #     obj = get_obj_leontief(p, x, budgets, valuations)
        #     objective_values.append(obj)
        # obj_hist_gda_leontief_all_high.append(objective_values)        
    
    return (prices_hist_gda_linear_all_low,
            demands_hist_gda_linear_all_low,
            obj_hist_gda_linear_all_low,
            prices_hist_gda_cd_all_low,
            demands_hist_gda_cd_all_low,
            obj_hist_gda_cd_all_low,
            prices_hist_gda_leontief_all_low,
            demands_hist_gda_leontief_all_low,
            obj_hist_gda_leontief_all_low,
            prices_hist_gda_linear_all_high,
            demands_hist_gda_linear_all_high,
            obj_hist_gda_linear_all_high,
            prices_hist_gda_cd_all_high,
            demands_hist_gda_cd_all_high,
            obj_hist_gda_cd_all_high,
            prices_hist_gda_leontief_all_high,
            demands_hist_gda_leontief_all_high,
            obj_hist_gda_leontief_all_high)


if __name__ == '__main__':

    num_experiments = 5
    num_buyers =  5
    num_goods = 8
    learning_rate_linear =  ((3,0.1), (1000**(-1/2),1000**(-1/2)))
    learning_rate_cd =  ((3,1), (500**(-1/2), 500**(-1/2)))
    learning_rate_leontief =  ((3,1), (500**(-1/2),500**(-1/2)))
    num_iters_linear = 2000
    num_iters_cd = 2000
    num_iters_leontief = 2000

    # results = 
    (prices_hist_gda_linear_all_low,
                demands_hist_gda_linear_all_low,
                obj_hist_gda_linear_all_low,
                prices_hist_gda_cd_all_low,
                demands_hist_gda_cd_all_low,
                obj_hist_gda_cd_all_low,
                prices_hist_gda_leontief_all_low,
                demands_hist_gda_leontief_all_low,
                obj_hist_gda_leontief_all_low,
                prices_hist_gda_linear_all_high,
                demands_hist_gda_linear_all_high,
                obj_hist_gda_linear_all_high,
                prices_hist_gda_cd_all_high,
                demands_hist_gda_cd_all_high,
                obj_hist_gda_cd_all_high,
                prices_hist_gda_leontief_all_high,
                demands_hist_gda_leontief_all_high,
                obj_hist_gda_leontief_all_high) = run_test(num_buyers, num_goods, learning_rate_linear, learning_rate_cd, learning_rate_leontief, num_experiments, num_iters_linear, num_iters_cd, num_iters_leontief)


    obj_hist_gda_linear_all_low = np.array(obj_hist_gda_linear_all_low)
    obj_hist_gda_cd_all_low = np.array(obj_hist_gda_cd_all_low)
    obj_hist_gda_leontief_all_low = np.array(obj_hist_gda_leontief_all_low)

    obj_hist_gda_linear_all_high = np.array(obj_hist_gda_linear_all_high)
    obj_hist_gda_cd_all_high = np.array(obj_hist_gda_cd_all_high)
    obj_hist_gda_leontief_all_high = np.array(obj_hist_gda_leontief_all_high)

    obj_hist_gda_linear_low =  pd.DataFrame( obj_hist_gda_linear_all_low)
    obj_hist_gda_cd_low =  pd.DataFrame(obj_hist_gda_cd_all_low)
    obj_hist_gda_leontief_low =  pd.DataFrame( obj_hist_gda_leontief_all_low)

    obj_hist_gda_linear_high =  pd.DataFrame( obj_hist_gda_linear_all_high)
    obj_hist_gda_cd_high =  pd.DataFrame(obj_hist_gda_cd_all_high)
    obj_hist_gda_leontief_high =  pd.DataFrame( obj_hist_gda_leontief_all_high)

    obj_hist_gda_linear_low.to_csv("data/obj/obj_hist_gda_linear_low.csv")
    obj_hist_gda_cd_low.to_csv("data/obj/obj_hist_gda_cd_low.csv")
    obj_hist_gda_leontief_low.to_csv("data/obj/obj_hist_gda_leontief_low.csv")

    obj_hist_gda_linear_high.to_csv("data/obj/obj_hist_gda_linear_high.csv")
    obj_hist_gda_cd_high.to_csv("data/obj/obj_hist_gda_cd_high.csv")
    obj_hist_gda_leontief_high.to_csv("data/obj/obj_hist_gda_leontief_high.csv")

    prices_hist_gda_linear_all_low = np.array(prices_hist_gda_linear_all_low)
    prices_hist_gda_cd_all_low = np.array(prices_hist_gda_cd_all_low)
    prices_hist_gda_leontief_all_low = np.array(prices_hist_gda_leontief_all_low)

    prices_hist_gda_linear_all_high = np.array(prices_hist_gda_linear_all_high)
    prices_hist_gda_cd_all_high = np.array(prices_hist_gda_cd_all_high)
    prices_hist_gda_leontief_all_high = np.array(prices_hist_gda_leontief_all_high)

    prices_gda_linear_low =  pd.DataFrame(prices_hist_gda_linear_all_low)
    prices_gda_cd_low =  pd.DataFrame(prices_hist_gda_cd_all_low)
    prices_gda_leontief_low =  pd.DataFrame( prices_hist_gda_leontief_all_low)

    prices_gda_linear_high =  pd.DataFrame(prices_hist_gda_linear_all_high)
    prices_gda_cd_high =  pd.DataFrame(prices_hist_gda_cd_all_high)
    prices_gda_leontief_high =  pd.DataFrame( prices_hist_gda_leontief_all_high)

    prices_gda_linear_low.to_csv("data/prices/prices_gda_linear_low.csv")
    prices_gda_cd_low.to_csv("data/prices/prices_gda_cd_low.csv")
    prices_gda_leontief_low.to_csv("data/prices/prices_gda_leontief_low.csv")

    prices_gda_linear_high.to_csv("data/prices/prices_gda_linear_high.csv")
    prices_gda_cd_high.to_csv("data/prices/prices_gda_cd_high.csv")
    prices_gda_leontief_high.to_csv("data/prices/prices_gda_leontief_high.csv")
    
    
    obj_gda_linear_low = np.mean(obj_hist_gda_linear_all_low, axis = 0)
    obj_gda_cd_low = np.mean(obj_hist_gda_cd_all_low, axis = 0)
    obj_gda_leontief_low = np.mean(obj_hist_gda_leontief_all_low, axis = 0)
    
    obj_gda_linear_low = obj_gda_linear_low - np.min(obj_gda_linear_low)
    obj_gda_cd_low = obj_gda_cd_low - np.min(obj_gda_cd_low)
    obj_gda_leontief_low = obj_gda_leontief_low - np.min(obj_gda_leontief_low)

    
    obj_gda_linear_high = np.mean(obj_hist_gda_linear_all_high, axis = 0)
    obj_gda_cd_high = np.mean(obj_hist_gda_cd_all_high, axis = 0)
    obj_gda_leontief_high = np.mean(obj_hist_gda_leontief_all_high, axis = 0)

    num_iters_linear = len(obj_gda_linear_low)
    num_iters_cd = len(obj_gda_cd_low)
    num_iters_leontief = len(obj_gda_leontief_low)
    x_linear = np.linspace(1, num_iters_linear, num_iters_linear)
    x_cd = np.linspace(1, num_iters_cd, num_iters_cd)
    x_leontief = np.linspace(1, num_iters_leontief, num_iters_leontief)

    plt.rcParams["font.size"] = 18

    fig, axs = plt.subplots(1, 3)

    axs[0].plot([iter for iter in range(num_iters_linear)], obj_gda_linear_low/(x_linear**(-1/2)), alpha = 1, color = "b")
    axs[0].set_title("Linear Market", fontsize = "medium")

    axs[1].plot([iter for iter in range(num_iters_cd)], obj_gda_cd_low/(x_cd**(-1/2)), color = "b")
    axs[1].set_title("Cobb-Douglas Market", fontsize = "medium")

    axs[2].plot([iter for iter in range(num_iters_leontief)], obj_gda_leontief_low/(x_leontief**(-1/2)), color = "b")
    axs[2].set_title("Leontief Market", fontsize = "medium")
    
    for ax in axs.flat:
        ax.set(xlabel='Iteration Number', ylabel=r'Explotability')
    #for ax in axs.flat:
    #    ax.label_outer()

    name = "obj_graphs"

    fig.set_size_inches(18.5, 5.5)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(f"graphs/{name}.jpg")
    plt.show()

