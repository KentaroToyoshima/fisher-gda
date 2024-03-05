from operator import index
import fisherMinmax as fm
import consumerUtility as cu
import numpy as np
import random
import os
import pandas as pd
import argparse
import time
from scipy.optimize import minimize
from plot import *
from utils import *
from references import References

parser = argparse.ArgumentParser()
parser.add_argument('-mt', '--market_types', nargs='+', choices=['linear', 'cd', 'leontief'], default=['leontief'])
parser.add_argument('-e', '--num_experiments', type=int, default=5)
parser.add_argument('-b', '--num_buyers', type=int, default=5)
parser.add_argument('-g', '--num_goods', type=int, default=8)
parser.add_argument('-li', '--learning_rate_linear', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('-cd', '--learning_rate_cd', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('-Le', '--learning_rate_leontief', nargs='+', type=float, default=[0.01, 0.01])
parser.add_argument('-mu', '--mutation_rate', nargs='+', type=float, default=[1, 1, 5])
parser.add_argument('-i', '--num_iters', type=int, default=5000)
parser.add_argument('-u', '--update_freq', type=int, default=500)
parser.add_argument('-a', '--arch', type=str, default='m-alg2', choices=['alg2', 'm-alg2', 'alg4'])
args = parser.parse_args()

def get_obj_value(prices, allocations, budgets, valuations, market_type):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer, :]
        if market_type == 'linear':
            utils[buyer] = cu.get_linear_indirect_util(prices.clip(min=0.0001), b, v)
        elif market_type == 'cd':
            utils[buyer] = cu.get_CD_indirect_util(prices.clip(min=0.0001), b, v)
        elif market_type == 'leontief':
            utils[buyer] = cu.get_leontief_indirect_util(prices.clip(min=0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min=0.0001))

#価格の平均を用いて目的関数を計算する関数
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
        allocations_0 = np.random.rand(num_buyers, num_goods)
        prices_0 = np.random.rand(num_goods) * 3 + 1
        bounds = [(0, None) for _ in prices_0]
        market_settings = {
            "linear": (valuations, learning_rate_linear, mutation_rate[0], allocations_linear_ref, prices_linear_ref),
            "cd": (valuations_cd, learning_rate_cd, mutation_rate[1], allocations_cd_ref, prices_cd_ref),
            "leontief": (valuations, learning_rate_leontief, mutation_rate[2], allocations_leontief_ref, prices_leontief_ref)
        }
        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")
        print(f"val = {valuations}\n budgets = {budgets}\n")
        for market_type in market_types:
            print(f"------ {market_type} Fisher Market ------")
            val, learning_rate, mutation, allocations_ref, prices_ref = market_settings[market_type]
            
            #results[market_type].append(run_experiment_time(fm.calc_gda, get_obj_value, market_type, num_buyers, val, budgets, allocations_0, prices_0, learning_rate, mutation, allocations_ref, prices_ref, num_iters, update_freq, arch))
            results[market_type].append(run_experiment(fm.calc_gda, get_obj_value, market_type, num_buyers, val, budgets, allocations_0, prices_0, learning_rate, mutation, allocations_ref, prices_ref, num_iters, update_freq, arch))
            objective_value[market_type].append(minimize(get_obj_value, prices_0, bounds=bounds, args=(allocations_0, budgets, val, market_type), method='SLSQP').fun)

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

    market_references = References()
    allocations_linear_ref = market_references.allocations_linear_ref2
    prices_linear_ref = market_references.prices_linear_ref2
    allocations_cd_ref = market_references.allocations_cd_ref2
    prices_cd_ref = market_references.prices_cd_ref2
    allocations_leontief_ref = market_references.allocations_leontief_ref2
    prices_leontief_ref = market_references.prices_leontief_ref2

    run_test(args.num_buyers, args.num_goods, allocations_linear_ref, allocations_cd_ref, allocations_leontief_ref, prices_linear_ref, prices_cd_ref, prices_leontief_ref, args.learning_rate_linear, args.learning_rate_cd, args.learning_rate_leontief, args.mutation_rate, args.num_experiments, args.num_iters, args.update_freq, args.arch, args.market_types, dir_obj, dir_allocations, dir_prices)

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
    plot_and_save_obj_graphs(obj_hist_data, plot_titles, args.market_types, dir_obj, dir_graphs, args.arch, objective_value)
    plot_and_save_prices_graphs(prices_hist_data, plot_titles, args.market_types, dir_prices, dir_graphs, args.arch)
    plot_and_save_allocations_graphs(plot_titles, args.market_types, dir_allocations, dir_graphs, args.arch, args.num_buyers)

if __name__ == '__main__':
    start = time.time()
    main(args)
    elapsed_time = time.time() - start
    
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")