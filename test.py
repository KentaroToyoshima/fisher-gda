
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
# TODO:pricesの推移をプロット
# TODO:demandの推移をプロット
# TODO:手法ごとにmutation rateやref strategyの値を変える
def run_test(num_buyers, num_goods, demands_linear_ref, demands_cd_ref, demands_leontief_ref, prices_linear_ref, prices_cd_ref, prices_leontief_ref, learning_rate_linear, learning_rate_cd, learning_rate_leontief, mutation_rate, num_experiments, num_iters):
    
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
        # Initialize random market parameters
        # NOTE:valuationとbudgetsを変えている
        valuations = np.random.rand(num_buyers, num_goods)*10 + 5
        valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T # Normalize valuations for Cobb-Douglas
        budgets = np.random.rand(num_buyers)*10 + 10
        # budgets = np.array([15.71,  14.916, 13.519, 13.967, 19.407])
        demands_0 = np.zeros(valuations.shape)
        prices_0  = np.random.rand(num_goods)*10 + 5

        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")
        print(f"****** Market Parameters ******\nval = {valuations}\n budgets = {budgets}\n")
        print(f"*******************************")
        print(f"------------ GDA ------------")

        print(f"------ Linear Fisher Market ------")
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_linear(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate_linear[0], mutation_rate, demands_linear_ref, prices_linear_ref, num_iters)
        
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
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_cd(num_buyers, valuations_cd, budgets, demands_0, prices_0, learning_rate_cd[0], mutation_rate, demands_cd_ref, prices_cd_ref, num_iters)
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
        #prices_0  = np.random.rand(num_goods) + 10
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_leontief(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate_leontief[0], mutation_rate, demands_leontief_ref, prices_leontief_ref, num_iters)
        prices_hist_gda_leontief_all_low.append(prices_gda)
        demands_hist_gda_leontief_all_low.append(demands_gda)
        objective_values = []
        for i in range(0, len(demands_hist_gda)):
            x = np.mean(np.array(demands_hist_gda[:i+1]).clip(min = 0), axis = 0)
            p = np.mean(np.array(prices_hist_gda[:i+1]).clip(min = 0), axis = 0)
            obj = get_obj_leontief(p, x, budgets, valuations)
            objective_values.append(obj)
        obj_hist_gda_leontief_all_low.append(objective_values)     
    
    #NOTE:各効用関数でdemand,price,objを保存している
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

    num_experiments = 50
    num_buyers = 5
    num_goods = 8
    learning_rate_linear =  ((2,0.1), (1000**(-1/2),1000**(-1/2)))  #(demand_lr, price_lr)
    learning_rate_cd =  ((2,0.1), (500**(-1/2), 500**(-1/2)))
    learning_rate_leontief =  ((2,0.1), (500**(-1/2),500**(-1/2)))
    mutation_rate = 1
    num_iters= 500

    #prices_0  = np.full(num_goods, 5, dtype=float)
    demands_linear_ref = np.array([[0.15870897253843697,0.18667388393005538,0.18262319641795074,0.23823041197502504,0.216446175522833,0.18785419361586111,0.23285530284893508,0.20507746522566062],
[0.2214629722042028,0.20860808797843736,0.22140787720885277,0.12582867717934845,0.21301308255590082,0.17276529955610048,0.24522152090120686,0.18522372158327838],
[0.20598304467483747,0.18441735271750875,0.177394288510474,0.2572936427742071,0.17507383048299144,0.17740357143142582,0.2578786629494514,0.17874099018154493],
[0.21133044246662247,0.23164742744370292,0.16653482755924817,0.21893520535502375,0.23551592001901875,0.22959460407602592,0.12318154087284972,0.20130681762206293],
[0.19931382866220138,0.18941322073160685,0.2480220856128333,0.16454023899668344,0.16201429058635855,0.23105622770410242,0.1434239927283041,0.2317231818367359]])
    prices_linear_ref = np.array([11.561, 11.72,  11.67,  11.247, 11.496, 11.587, 11.433, 11.412])

    demands_cd_ref = np.array([[0.1995556914054147,0.20165524477521304,0.19931980223021523,0.21632457826934778,0.20787097054079648,0.20039455450426125,0.2105604994073406,0.1938437875510947],
[0.2041706251959548,0.2026957637286098,0.2101141903995081,0.18036968807710924,0.20207163553748636,0.19497829771435835,0.20384670607543312,0.19860233725220827],
[0.20161655204518536,0.20518956524621643,0.19713009601735923,0.21690380709960821,0.1974723146980503,0.2106600686225747,0.20725089619422657,0.20592135061060687],
[0.19514304080621675,0.2000556540932039,0.19788390292250566,0.19621578711161786,0.19972305547030714,0.19996478878803362,0.19595274477178917,0.2027797062542929],
[0.19951409054722855,0.19040377215675672,0.19555200843041173,0.19018613944231672,0.19286202375335956,0.19400229037077227,0.18238915355121046,0.19885281833179747]])
    prices_cd_ref = np.array([9.412, 9.289, 9.58,  9.127, 9.659, 9.425, 9.598, 9.434])

    demands_leontief_ref = np.array([[0.19705567341885252,0.1982915575355722,0.19773807989732078,0.21409476185120305,0.20372537548710448,0.2010255929828345,0.21065061187174436,0.1905887832707092],
[0.1992753959495068,0.19548523136325677,0.20307522600549718,0.17542932866144834,0.19647843442865903,0.18795989989276596,0.19501276061483022,0.1915799048637715],
[0.19424500982922424,0.19621123887398068,0.18764810592723333,0.2024828158026641,0.18337615275460595,0.19881062937452743,0.19567830874069533,0.19343444099148158],
[0.19383242365257075,0.19887022478796612,0.19875305914565522,0.19467848421904685,0.1976601041748803,0.1985466548518853,0.19833681676842804,0.20356241608579181],
[0.19435387768807794,0.1859283143415589,0.19023001142886942,0.18792336030600126,0.18723355777594416,0.18994284934439837,0.1782275204450602,0.19480714304994362]])
    prices_leontief_ref = np.array([10.55,   8.292, 10.686, 10.273,  6.996,  9.43,  11.713,  7.312])

    # results
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
                obj_hist_gda_leontief_all_high) = run_test(num_buyers, num_goods,
                                                            demands_linear_ref, demands_cd_ref, demands_leontief_ref, 
                                                            prices_linear_ref, prices_cd_ref, prices_leontief_ref, 
                                                            learning_rate_linear, learning_rate_cd, learning_rate_leontief, mutation_rate, num_experiments, num_iters)

    # save results
    obj_hist_gda_linear_all_low = np.array(obj_hist_gda_linear_all_low)
    obj_hist_gda_cd_all_low = np.array(obj_hist_gda_cd_all_low)
    obj_hist_gda_leontief_all_low = np.array(obj_hist_gda_leontief_all_low)

    obj_hist_gda_linear_low =  pd.DataFrame( obj_hist_gda_linear_all_low)
    obj_hist_gda_cd_low =  pd.DataFrame(obj_hist_gda_cd_all_low)
    obj_hist_gda_leontief_low =  pd.DataFrame( obj_hist_gda_leontief_all_low)

    obj_hist_gda_linear_low.to_csv("data/obj/obj_hist_gda_linear_low.csv")
    obj_hist_gda_cd_low.to_csv("data/obj/obj_hist_gda_cd_low.csv")
    obj_hist_gda_leontief_low.to_csv("data/obj/obj_hist_gda_leontief_low.csv")

    prices_hist_gda_linear_all_low = np.array(prices_hist_gda_linear_all_low)
    prices_hist_gda_cd_all_low = np.array(prices_hist_gda_cd_all_low)
    prices_hist_gda_leontief_all_low = np.array(prices_hist_gda_leontief_all_low)

    prices_gda_linear_low =  pd.DataFrame(prices_hist_gda_linear_all_low)
    prices_gda_cd_low =  pd.DataFrame(prices_hist_gda_cd_all_low)
    prices_gda_leontief_low =  pd.DataFrame( prices_hist_gda_leontief_all_low)

    prices_gda_linear_low.to_csv("data/prices/prices_gda_linear_low.csv")
    prices_gda_cd_low.to_csv("data/prices/prices_gda_cd_low.csv")
    prices_gda_leontief_low.to_csv("data/prices/prices_gda_leontief_low.csv")

    print(np.mean(prices_hist_gda_linear_all_low, axis=0))
    print(np.mean(prices_hist_gda_cd_all_low, axis=0))
    print(np.mean(prices_hist_gda_leontief_all_low, axis=0))

    demands_hist_gda_linear_all_low = np.array(np.mean(demands_hist_gda_linear_all_low, axis=0))
    demands_hist_gda_cd_all_low = np.array(np.mean(demands_hist_gda_cd_all_low, axis=0))
    demands_hist_gda_leontief_all_low = np.array(np.mean(demands_hist_gda_leontief_all_low, axis=0))

    demands_gda_linear_low =  pd.DataFrame(demands_hist_gda_linear_all_low)
    demands_gda_cd_low =  pd.DataFrame(demands_hist_gda_cd_all_low)
    demands_gda_leontief_low =  pd.DataFrame(demands_hist_gda_leontief_all_low)

    demands_gda_linear_low.to_csv("data/demands/demands_gda_linear_low.csv")
    demands_gda_cd_low.to_csv("data/demands/demands_gda_cd_low.csv")
    demands_gda_leontief_low.to_csv("data/demands/demands_gda_leontief_low.csv")

    obj_gda_linear_low = np.mean(obj_hist_gda_linear_all_low, axis = 0)
    obj_gda_cd_low = np.mean(obj_hist_gda_cd_all_low, axis = 0)
    obj_gda_leontief_low = np.mean(obj_hist_gda_leontief_all_low, axis = 0)
    
    obj_gda_linear_low = obj_gda_linear_low - np.min(obj_gda_linear_low)
    obj_gda_cd_low = obj_gda_cd_low - np.min(obj_gda_cd_low)
    obj_gda_leontief_low = obj_gda_leontief_low - np.min(obj_gda_leontief_low)

    # plot
    num_iters_linear = len(obj_gda_linear_low)
    num_iters_cd = len(obj_gda_cd_low)
    num_iters_leontief = len(obj_gda_leontief_low)
    # x_linear = np.linspace(1, num_iters_linear, num_iters_linear)
    # x_cd = np.linspace(1, num_iters_cd, num_iters_cd)
    # x_leontief = np.linspace(1, num_iters_leontief, num_iters_leontief)

    fig, axs = plt.subplots(1, 3)
    
    # Add shift in plots to make the difference clearer
    axs[0].plot([iter for iter in range(num_iters_linear)], obj_gda_linear_low, color = "b")
    axs[0].set_title("Linear Market", fontsize = "medium")

    axs[1].plot([iter for iter in range(num_iters_cd)], obj_gda_cd_low, color = "b")
    axs[1].set_title("Cobb-Douglas Market", fontsize = "medium")

    axs[2].plot([iter for iter in range(num_iters_leontief)], obj_gda_leontief_low, color = "b")
    axs[2].set_title("Leontief Market", fontsize = "medium")
    
    for ax in axs.flat:
        ax.set(xlabel='Iteration Number', ylabel=r'Explotability')

    name = "obj_graphs"

    fig.set_size_inches(18.5, 5.5)
    plt.rcParams["font.size"] = 18
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(f"graphs/{name}.jpg")
    plt.show()

