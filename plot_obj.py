
import fisherMinmax as fm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import consumerUtility as cu
from datetime import date
import glob

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
def run_test(num_buyers, num_goods, demands_linear_ref, demands_cd_ref, demands_leontief_ref, prices_linear_ref, prices_cd_ref, prices_leontief_ref, learning_rate_linear, learning_rate_cd, learning_rate_leontief, mutation_rate, num_experiments, num_iters, update_num, arch):
    
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
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_linear(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate_linear[0], mutation_rate, demands_linear_ref, prices_linear_ref, num_iters, update_num, arch)
        
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
        
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_cd(num_buyers, valuations_cd, budgets, demands_0, prices_0, learning_rate_cd[0], mutation_rate, demands_cd_ref, prices_cd_ref, num_iters, update_num, arch)
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
        demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_leontief(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate_leontief[0], mutation_rate, demands_leontief_ref, prices_leontief_ref, num_iters, update_num, arch)
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
    #file_ = glob.glob('{}/csv/trajectory_{}.csv'.format(dir_path, i_s))[0]
    #df_alg4_linear = glob.glob('results/2022_12_26_04_03_28_408959_alg4_en_10_iters_500_ml_[1, 1, 1]_uf_0/data/prices/alg4_*_linear_[0-9].csv')
    #df_alg4_cd = glob.glob('results/2022_12_26_04_03_28_408959_alg4_en_10_iters_500_ml_[1, 1, 1]_uf_0/data/prices/alg4_*_cd_[0-9].csv')
    df_alg4_leontief = glob.glob('results/2022_12_26_04_03_28_408959_alg4_en_10_iters_500_ml_[1, 1, 1]_uf_0/data/prices/alg4_*_leontief_[0-9].csv')
    #df_m_alg2_linear = pd.read_csv('data/obj/m-alg2_exploit_gda_linear_low.csv', index_col=0)
    #df_m_alg2_cd = pd.read_csv('data/obj/m-alg2_exploit_gda_cd_low.csv', index_col=0)
    #df_m_alg2_leontief = pd.read_csv('data/obj/m-alg2_exploit_gda_leontief_low.csv', index_col=0)
    
    #print(obj_gda_linear_low)
    #print(obj_gda_cd_low)
    #print(obj_gda_leontief_low)

    # plot
    # x_linear = np.linspace(1, num_iters_linear, num_iters_linear)
    # x_cd = np.linspace(1, num_iters_cd, num_iters_cd)
    # x_leontief = np.linspace(1, num_iters_leontief, num_iters_leontief)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 15))

    #df_alg4_linear.plot(ax=axes[0,0])
    #df_alg4_cd.plot(ax=axes[0,1])
    df_alg4_leontief.plot(ax=axes[0,2])
    #df_m_alg2_linear.plot(ax=axes[1,0])
    #df_m_alg2_cd.plot(ax=axes[1,1])
    #df_m_alg2_leontief.plot(ax=axes[1,2])

    for i in range(1):
        for j in range(3):
            axes[i,j].set_ylim(-0.1, 1)
            axes[i,j].tick_params(axis='x', labelsize=15)
            axes[i,j].tick_params(axis='y', labelsize=15)
            axes[i,j].grid(True)

    #axes[0, 0].set_title("Linear Market", fontsize=20)
    #axes[0, 1].set_title("Cobb-Douglas Martket", fontsize=20)
    #axes[0, 2].set_title("Leontief Martket", fontsize=20)
    #axes[0, 0].set_ylabel("Exploitability", fontsize=20)
    #axes[1, 0].set_ylabel("Exploitability", fontsize=20)
    #axes[1, 0].set_xlabel("Iterations", fontsize=20)
    #axes[1, 1].set_xlabel("Iterations", fontsize=20)
    #axes[1, 2].set_xlabel("Iterations", fontsize=20)
    plt.show()

    plt.rcParams["font.size"] = 18
    plt.savefig(f"graphs/aaa_obj_graphs.jpg")
