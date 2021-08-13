
import fisherMinmax as fm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import consumerUtility as cu
from datetime import date

# Objective Functions for linear, Cobb-Douglas, and Leontief

def get_obj_linear(prices, demands, budgets, valuations):
    utils = np.sum(valuations*demands, axis = 1)
    return np.sum(prices) + budgets.T @ np.log(utils)

def get_obj_cd(prices, demands, budgets, valuations):
    utils = np.prod(np.power(demands, valuations), axis= 1)

    return np.sum(prices) + budgets.T @ np.log(utils)

def get_obj_leontief(prices, demands, budgets, valuations):
    utils = np.min(demands/valuations, axis = 1)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001))

# Function that run Max-Oracle Gradient Descent and Nested Gradient Descent Ascent Tests and returns data
def run_test(num_buyers, num_goods, learning_rate, num_experiments, num_iters_linear , num_iters_cd, num_iters_leontief):
    
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
        budgets = np.random.rand(num_buyers)*10 + 100
        
        
        print(f"****** Market Parameters ******\nval = {valuations}\n budgets = {budgets}\n")
        print(f"*******************************")
        
        print(f"------------ Low Initial Prices ------------")
        
        print(f"------ Linear Fisher Market ------\n")
        prices_0  = np.random.rand(num_goods)*10 + 5
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_linear(valuations, budgets, prices_0, learning_rate, num_iters_linear)
        
        prices_hist_gda_linear_all_low.append(prices_gda)
        demands_hist_gda_linear_all_low.append(demands_gda)
        objective_values = []
        for x, p in zip(demands_hist_gda, prices_hist_gda):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_gda_linear_all_low.append(objective_values)
        
        
        prices_0  = np.random.rand(num_goods) + 5
        print(f"------ Cobb-Douglas Fisher Market ------")
        valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T # Normalize valuations for Cobb-Douglas
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_cd)
        prices_hist_gda_cd_all_low.append(prices_gda)
        demands_hist_gda_cd_all_low.append(demands_gda)
        objective_values = []
        
        for x, p in zip(demands_hist_gda, prices_hist_gda):
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_gda_cd_all_low.append(objective_values)
        
        print(f"------ Leontief Fisher Market ------")
        
        prices_0  = np.random.rand(num_goods) + 5
        
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_leontief(valuations, budgets, prices_0, learning_rate, num_iters_leontief)
        prices_hist_gda_leontief_all_low.append(prices_gda)
        demands_hist_gda_leontief_all_low.append(demands_gda)
        objective_values = []
        for x, p in zip(demands_hist_gda, prices_hist_gda):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_gda_leontief_all_low.append(objective_values)        
        

        print(f"------------ High Initial Prices ------------")

        print(f"------ Linear Fisher Market ------\n")
        
        prices_0  = np.random.rand(num_goods)*5  + 50
        print(f"***** Max-Oracle Gradient Descent *****")
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_linear(valuations, budgets, prices_0, learning_rate, num_iters_linear)
        
        prices_hist_gda_linear_all_high.append(prices_gda)
        demands_hist_gda_linear_all_high.append(demands_gda)
        objective_values = []
        for x, p in zip(demands_hist_gda, prices_hist_gda):
            obj = get_obj_linear(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_gda_linear_all_high.append(objective_values)
        
        print(f"------ Cobb-Douglas Fisher Market ------")
        
        # Normalize valuations for Cobb-Douglas
        valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T 
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_cd(valuations_cd, budgets, prices_0, learning_rate, num_iters_cd)
        prices_hist_gda_cd_all_high.append(prices_gda)
        demands_hist_gda_cd_all_high.append(demands_gda)
        objective_values = []
        
        for x, p in zip(demands_hist_gda, prices_hist_gda):
            obj = get_obj_cd(p, x, budgets, valuations_cd)
            objective_values.append(obj)
        obj_hist_gda_cd_all_high.append(objective_values)
        
        print(f"------ Leontief Fisher Market ------")
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_leontief(valuations, budgets, prices_0, learning_rate, num_iters_leontief)
        prices_hist_gda_leontief_all_high.append(prices_gda)
        demands_hist_gda_leontief_all_high.append(demands_gda)
        objective_values = []
        for x, p in zip(demands_hist_gda, prices_hist_gda):
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_gda_leontief_all_high.append(objective_values)        

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

    num_experiments = 1
    num_buyers =  5
    num_goods = 8
    learning_rate =  5
    num_iters_linear = 250
    num_iters_cd = 250
    num_iters_leontief = 250

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
                obj_hist_gda_leontief_all_high) = run_test(num_buyers, num_goods, learning_rate, num_experiments, num_iters_linear, num_iters_cd, num_iters_leontief)


    # # # Save    obj_hist_gda_linear_all_low = np.array(obj_hist_gda_linear_all_low)
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

    obj_gda_linear_high = np.mean(obj_hist_gda_linear_all_high, axis = 0)
    obj_gda_cd_high = np.mean(obj_hist_gda_cd_all_high, axis = 0)
    obj_gda_leontief_high = np.mean(obj_hist_gda_leontief_all_high, axis = 0)

    # obj_gda_leontief_low = obj_gda_leontief_low[:-200]
    # obj_gda_leontief_high = obj_gda_leontief_high[:-200]

    num_iters_linear = len(obj_gda_linear_low)
    num_iters_cd = len(obj_gda_cd_low)
    num_iters_leontief = len(obj_gda_leontief_low)
    x_linear = np.linspace(1, num_iters_linear, num_iters_linear)
    x_cd = np.linspace(1, num_iters_cd, num_iters_cd)
    x_leontief = np.linspace(1, num_iters_leontief, num_iters_leontief)

    fig, axs = plt.subplots(2, 3) # Create a figure containing a single axes.
    # First row for experiments with low initial prices and
    # second row for experiments with high initial prices.

    # Add shift in plots to make the difference clearer
    axs[0,0].plot([iter for iter in range(num_iters_linear)], obj_gda_linear_low, label = "Max-Oracle", alpha = 1, color = "b")
    # axs[0,0].p (obj_gda_linear[0]/15)/x + obj_gda_linear[-1], color='green', linestyle='dashed', label = "1/T")
    axs[0,0].set_title("Linear Market", fontsize = "medium")
    # axs[0,0].set_ylim(2100, 2500)

    axs[0,1].plot([iter for iter in range(num_iters_cd)], obj_gda_cd_low , label = "Max-Oracle", color = "b")
    # axs[0,1].plot(x, (obj_gda_cd[0]/3)/x + obj_gda_cd[-1], color='green', linestyle='dashed', label = "1/T")
    # axs[0,1].plot(x_cd, (obj_gda_cd_low[0] - obj_gda_cd_low[-1])*(x_cd**(-1/2)) + obj_gda_cd_low[-1], color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[0,1].set_title("Cobb-Douglas Market", fontsize = "medium")
    # axs[0,1].set_ylim(-330, 200)

    axs[0,2].plot([iter for iter in range(num_iters_leontief)], obj_gda_leontief_low, label = "Max-Oracle", color = "b")
    # axs[1,0].plot(x, (obj_gda_leontief[0]/4)/x + obj_gda_leontief[-1], color='green', linestyle='dashed', label = "1/T")
    # axs[0,2].plot(x_leontief, (obj_gda_leontief_low[0] - obj_gda_leontief_low[-1])*(x_leontief**(-1/2)) + obj_gda_leontief_low[-1], color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[0,2].set_title("Leontief Market", fontsize = "medium")
    # axs[1, 1].axis('off')
    # axs[0,2].set_ylim(-1600, -900)

    # Add shift in plots to make the difference clearer
    axs[1,0].plot([iter for iter in range(num_iters_linear)], obj_gda_linear_high, label = "Max-Oracle", alpha = 1, color = "b")
    # axs[0,0].p (obj_gda_linear[0]/15)/x + obj_gda_linear[-1], color='green', linestyle='dashed', label = "1/T")
    axs[1,0].set_title("Linear Market", fontsize = "medium")
    # axs[1,0].set_ylim(2115, 2145)

    axs[1,1].plot([iter for iter in range(num_iters_cd)], obj_gda_cd_high , label = "Max-Oracle", alpha = 1, color = "b")
    # axs[0,1].plot(x, (obj_gda_cd[0]/3)/x + obj_gda_cd[-1], color='green', linestyle='dashed', label = "1/T")
    # axs[1,1].plot(x_cd, (obj_gda_cd_high[0] - obj_gda_cd_high[-1])*(x_cd**(-1/2)) + obj_gda_cd_high[-1] , color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[1,1].set_title("Cobb-Douglas Market", fontsize = "medium")
    # axs[1,1].set_ylim(-305, -290)

    axs[1,2].plot([iter for iter in range(num_iters_leontief)], obj_gda_leontief_high, label = "Max-Oracle", alpha = 1, color = "b")
    # axs[1,0].plot(x, (obj_gda_leontief[0]/4)/x + obj_gda_leontief[-1], color='green', linestyle='dashed', label = "1/T")
    # axs[1,2].plot(x_leontief, (obj_gda_leontief_high[0] - obj_gda_leontief_high[-1])*(x_leontief**(-1/2)) + obj_gda_leontief_high[-1] - 3, color='red', linestyle='dashed', label = "1/sqrt(T)")
    axs[1,2].set_title("Leontief Market", fontsize = "medium")


    for ax in axs.flat:
        ax.set(xlabel='Iteration Number', ylabel='Objective Value')
        ax.yaxis.set_ticks([])
    for ax in axs.flat:
        ax.label_outer()

    name = "obj_graphs"
    plt.savefig(f"graphs/{name}.jpg")
    plt.show()

