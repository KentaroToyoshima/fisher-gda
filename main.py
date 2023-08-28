from operator import index
import fisherMinmax as fm
import consumerUtility as cu
import numpy as np
import random
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import glob
from pathlib import Path
import re
import json
import argparse
import time
from scipy.optimize import minimize
from plot import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-mt', '--market_types', nargs='+', choices=['linear', 'cd', 'leontief'], default=['linear', 'cd', 'leontief'])
parser.add_argument('-e', '--num_experiments', type=int, default=5)
parser.add_argument('-b', '--num_buyers', type=int, default=5)
parser.add_argument('-g', '--num_goods', type=int, default=8)
parser.add_argument('-li', '--learning_rate_linear', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('-cd', '--learning_rate_cd', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('-Le', '--learning_rate_leontief', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('-mu', '--mutation_rate', nargs='+', type=float, default=[1, 1, 1])
parser.add_argument('-i', '--num_iters', type=int, default=1000)
parser.add_argument('-u', '--update_freq', type=int, default=0)
parser.add_argument('-a', '--arch', type=str, default='alg4', choices=['alg2', 'm-alg2', 'alg4'])
args = parser.parse_args()

# Objective Functions for linear, Cobb-Douglas, and Leontief
def get_obj_value(prices, allocations, budgets, valuations, market_type):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer, :]
        if market_type == 'linear':
            utils[buyer] = cu.get_linear_indirect_utill(prices.clip(min=0.0001), b, v)
        elif market_type == 'cd':
            utils[buyer] = cu.get_CD_indirect_util(prices.clip(min=0.0001), b, v)
        elif market_type == 'leontief':
            utils[buyer] = cu.get_leontief_indirect_util(prices.clip(min=0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min=0.0001))

def run_experiment_time_average(fm_func, get_obj_value, market_type, num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch):
    allocations_gda, prices_gda, allocations_hist_gda, prices_hist_gda = fm_func(num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch, market_type)

    prices_hist_array = np.array(prices_hist_gda)
    average_price_list = []
    for i in range(1, len(prices_hist_array) + 1):
        average_price = np.mean(prices_hist_array[:i], axis=0)
        average_price_list.append(average_price)
    objective_values = [get_obj_value(p, x, budgets, valuations, market_type) for p, x in zip(average_price_list, allocations_hist_gda)]
    return allocations_hist_gda, prices_hist_gda, objective_values

def run_experiment(fm_func, get_obj_value, market_type, num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch):
    allocations_gda, prices_gda, allocations_hist_gda, prices_hist_gda = fm_func(num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch, market_type)
    objective_values = [get_obj_value(p, x, budgets, valuations ,market_type) for p, x in zip(prices_hist_gda, allocations_hist_gda)]
    return allocations_hist_gda, prices_hist_gda, objective_values

def run_test(num_buyers, num_goods, allocations_linear_ref, allocations_cd_ref, allocations_leontief_ref, prices_linear_ref, prices_cd_ref, prices_leontief_ref, learning_rate_linear, learning_rate_cd, learning_rate_leontief, mutation_rate, num_experiments, num_iters, update_freq, arch, market_types, dir_obj, dir_allocations, dir_prices):
    results = {key: [] for key in market_types}
    global objective_value
    objective_value = {key: [] for key in market_types}

    for experiment_num in range(num_experiments):
        np.random.seed(experiment_num)
        random.seed(experiment_num)
        valuations = np.random.rand(num_buyers, num_goods) * 10 + 5
        valuations_cd = (valuations.T / np.sum(valuations, axis=1)).T
        budgets = np.random.rand(num_buyers) * 10 + 10
        allocations_0 = np.random.rand(num_buyers, num_goods) + 1
        prices_0 = np.random.rand(num_goods)
        bounds = [(0, None) for _ in prices_0]
        market_settings = {
            "linear": (valuations, learning_rate_linear, mutation_rate[0], allocations_linear_ref, prices_linear_ref),
            "cd": (valuations_cd, learning_rate_cd, mutation_rate[1], allocations_cd_ref, prices_cd_ref),
            "leontief": (valuations, learning_rate_leontief, mutation_rate[2], allocations_leontief_ref, prices_leontief_ref)
        }
        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")
        print(f"val = {valuations}\n budgets = {budgets}\n")
        for market_type in market_types:
        #TODO:time_averageと普通のやつを分けて指定できるようにする
            print(f"------ {market_type} Fisher Market ------")
            val, learning_rate, mutation, allocations_ref, prices_ref = market_settings[market_type]
            
            #results[market_type].append(run_experiment_time(fm.calc_gda, get_obj_value, market_type, num_buyers, val, budgets, allocations_0, prices_0, learning_rate, mutation, allocations_ref, prices_ref, num_iters, update_freq, arch))
            results[market_type].append(run_experiment(fm.calc_gda, get_obj_value, market_type, num_buyers, val, budgets, allocations_0, prices_0, learning_rate, mutation, allocations_ref, prices_ref, num_iters, update_freq, arch))
            objective_value[market_type].append(minimize(get_obj_value, prices_0, bounds=bounds, args=(allocations_0, budgets, val, market_type), method='TNC').fun)

        # Save data
        for key in results:
            allocations_hist, prices_hist, obj_hist = zip(*results[key])

            allocations_hist_all = np.array(allocations_hist[-1]).swapaxes(0, 1)
            prices_hist_all = np.array(prices_hist[-1])
            obj_hist_all = np.array(obj_hist[-1])

            # Save prices
            df_prices = pd.DataFrame(prices_hist_all)
            df_prices.to_csv(f"{dir_prices}/{arch}_prices_hist_{key}_{experiment_num}.csv")

            # Save allocations
            for buyer in range(num_buyers):
                df_allocations = pd.DataFrame(allocations_hist_all[buyer])
                df_allocations.to_csv(f"{dir_allocations}/{arch}_allocations_hist_{key}_{experiment_num}_buyer_{buyer}.csv")

            # Save objective function
            df_obj = pd.DataFrame(obj_hist_all)
            df_obj.to_csv(f"{dir_obj}/{arch}_obj_hist_{key}_{experiment_num}.csv")

def main(args):
    dir_data, dir_obj, dir_allocations, dir_prices, dir_graphs = create_directories(args.arch, args.num_experiments, args.num_iters, args.update_freq)

    write_params_to_file(args.market_types, args.num_experiments, args.num_buyers, args.num_goods, args.learning_rate_linear, args.learning_rate_cd, args.learning_rate_leontief, args.mutation_rate, args.num_iters, args.update_freq, args.arch, dir_data)
    #TODO:収束先を実験のフォルダを指定するだけで定義できるようにする
    #収束先
    allocations_linear_ref2 = np.array([[0.2140533516657123,0.3061049563869186,0.27166908707005966,0.09571924111289291,0.0901534275339205,0.1871724245988343,0.2388888704801567,0.17565218791642462],
    [0.19524636955165617,0.22399845066806826,0.1605925706417905,0.24230411054588,0.3008993418189746,0.16602469940876047,0.1674345879496748,0.23229037423201512],
    [0.2627728479780005,0.09143493544940536,0.32627628167202094,0.11954669073171782,0.21691127438984087,0.32869768533375454,0.23589615246925652,0.10651353133012238],
    [0.10539172895354212,0.1961667008995463,0.09940269338520986,0.2804478412546389,0.17582705778263175,0.1666113991243431,0.26783814219447943,0.29808539069440976],
    [0.2520448172229872,0.1782818927609587,0.10388683374179257,0.24669910164714598,0.23240726975797948,0.17174820153877457,0.1596453960270177,0.1714572420480117]])
    prices_linear_ref2 = np.array([10.305009236665587,9.867552078184502,9.765631740425116,10.503610801252867,10.484207649086361,10.155319357469878,10.06851615467792,10.114892830529984])
    allocations_linear_ref = np.full_like(allocations_linear_ref2, 0.01)
    prices_linear_ref = np.full_like(prices_linear_ref2, 5)
    #allocations_linear_ref = np.random.rand(*allocations_linear_ref2.shape) + 0.1
    #prices_linear_ref = np.random.rand(*prices_linear_ref2.shape)*5
    #収束先
    allocations_cd_ref2 = np.array([[0.19882207346959496,0.20454969041013799,0.20301853791617144,0.18350679664792788,0.1848263104906626,0.18776534860985633,0.208349163569394,0.18270868999562612],
    [0.20511539324555583,0.20183085431590692,0.2227389942634766,0.21795332966637165,0.22514749605715467,0.20514626859681692,0.20115062832681754,0.21874227275737973],
    [0.1916512561780199,0.20479231693757433,0.20055768285824555,0.1917134249899315,0.21031217155416812,0.20873926276662164,0.20508245039446735,0.20350010150683548],
    [0.19867563522246953,0.20308858101617866,0.19485206609760053,0.21784490422507607,0.18558821395318137,0.21780528060932497,0.19931167738756894,0.21167614148273586],
    [0.20747429147055482,0.1876836062006191,0.1806401069787935,0.19117202699863203,0.1963349739329846,0.18279746454826035,0.18756410239004,0.18577336361679564]])
    prices_cd_ref2 = np.array([8.919804195898308,9.083935664624336,8.931057250091445,9.255373557174382,9.428494518784056,9.508329733039915,8.702703697870284,9.402731490182411])
    allocations_cd_ref = np.full_like(allocations_cd_ref2, 0.01)
    prices_cd_ref = np.full_like(prices_cd_ref2, 5)
    #allocations_cd_ref = np.random.rand(*allocations_cd_ref2.shape) + 0.1
    #prices_cd_ref = np.random.rand(*prices_cd_ref2.shape)*5
    #収束先
    allocations_leontief_ref2 = np.array([[0.12866456874290486,0.18544965057375923,0.15668769709597827,0.2788845306987129,0.26668936860283254,0.12997920374110544,0.21050759357411178,0.19361244009502113],
    [0.1681461228769793,0.24378126805301706,0.21749649338531438,0.14355888559827204,0.2187185913728465,0.21081826828905736,0.19868437404216674,0.29914133559530093],
    [0.2727733867992524,0.20427932832241838,0.20619830345926266,0.2869555750062897,0.14732658456364273,0.19226849018590775,0.1340008015079105,0.17044016955142247],
    [0.23462187856827854,0.1636721017430995,0.21981513033975159,0.1560446702010896,0.2246448731638046,0.20863863957370196,0.2599558057820761,0.16525922115090244],
    [0.17115167200961734,0.1913316969652631,0.19916199827895448,0.13210022527618612,0.15419022886931527,0.24289971587504913,0.2115231787158105,0.2133065056036007]])
    prices_leontief_ref2 = np.array([9.160170331214566,9.158025644512737,9.159421329401761,9.159060411237837,9.157849830207363,9.158670796130883,9.158345305893103,9.158696747562974])
    allocations_leontief_ref = np.full_like(allocations_leontief_ref2, 0.01)
    prices_leontief_ref = np.full_like(prices_leontief_ref2, 5)
    #allocations_leontief_ref = np.random.rand(*allocations_leontief_ref2.shape) + 0.1
    #prices_leontief_ref = np.random.rand(*prices_leontief_ref2.shape)*5
    # results
    run_test(args.num_buyers, args.num_goods, allocations_linear_ref, allocations_cd_ref, allocations_leontief_ref, prices_linear_ref, prices_cd_ref, prices_leontief_ref, args.learning_rate_linear, args.learning_rate_cd, args.learning_rate_leontief, args.mutation_rate, args.num_experiments, args.num_iters, args.update_freq, args.arch, args.market_types, dir_obj, dir_allocations, dir_prices)

    #TODO:以下をもっと綺麗に書く（objとpriceはdataはここでデータを取ってきているのに、allocationsはplot関数で取ってきている）
    patterns = [rf'.*{key}.*\.csv' for key in args.market_types]
    dir_content = os.listdir(dir_obj)
    obj_hist_data = [get_dataframes(pattern, dir_content, dir_obj) for pattern in patterns]
    dir_content = os.listdir(dir_prices)
    prices_hist_data = [get_dataframes(pattern, dir_content, dir_prices) for pattern in patterns]
    plot_titles_dict = {
        'linear': "Linear Market",
        'cd': "Cobb-Douglas Market",
        'leontief': "Leontief Market"
    }
    plot_titles = [plot_titles_dict[market] for market in args.market_types]
    #plot_and_save_obj_graphs(obj_hist_data, plot_titles, args.market_types, dir_obj, dir_graphs, args.arch)
    plot_and_save_obj_graphs_followed_paper(obj_hist_data, plot_titles, args.market_types, dir_obj, dir_graphs, args.arch, objective_value)
    plot_and_save_prices_graphs(prices_hist_data, plot_titles, args.market_types, dir_prices, dir_graphs, args.arch)
    plot_and_save_allocations_graphs(plot_titles, args.market_types, dir_allocations, dir_graphs, args.arch, args.num_buyers)

if __name__ == '__main__':
    start = time.time()
    main(args)
    elapsed_time = time.time() - start
    
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")