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

# Objective Functions for linear, Cobb-Douglas, and Leontief
def get_obj_linear(prices, allocations, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_linear_indirect_utill(prices.clip(min= 0.0001), b, v)
        #utils[buyer] = cu.get_linear_utility(allocations[buyer,:], v)
    # return np.sum(prices) + budgets.T @ np.log(utils) + np.sum(budgets) - np.sum(allocations @ prices )
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min = 0.0001))

def get_obj_cd(prices, allocations, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_CD_indirect_util(prices.clip(min= 0.0001), b, v)
        #utils[buyer] = cu.get_CD_utility(allocations[buyer,:], v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001)) 

def get_obj_leontief(prices, allocations, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_leontief_indirect_util(prices.clip(min= 0.0001), b, v)
        #utils[buyer] = cu.get_leontief_utility(allocations[buyer,:], v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001))

# Function that run Max-Oracle Gradient Descent and Nested Gradient Descent Ascent Tests and returns data
def run_experiment_time_average(fm_func, get_obj, market_type, num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch):
    allocations_gda, prices_gda, allocations_hist_gda, prices_hist_gda = fm_func(num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch, market_type)

    prices_hist_array = np.array(prices_hist_gda)
    average_price_list = []
    for i in range(1, len(prices_hist_array) + 1):
        average_price = np.mean(prices_hist_array[:i], axis=0)
        average_price_list.append(average_price)
    objective_values = [get_obj(p, x, budgets, valuations) for p, x in zip(average_price_list, allocations_hist_gda)]
    return allocations_hist_gda, prices_hist_gda, objective_values

def run_experiment(fm_func, get_obj, market_type, num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch):
    allocations_gda, prices_gda, allocations_hist_gda, prices_hist_gda = fm_func(num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate, mutation_rate, allocations_ref, prices_ref, num_iters, update_freq, arch, market_type)
    objective_values = [get_obj(p, x, budgets, valuations) for p, x in zip(prices_hist_gda, allocations_hist_gda)]
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
        allocations_0 = np.random.rand(num_buyers, num_goods) + 0.1
        prices_0 = np.random.rand(num_goods)
        bounds = [(0, None) for _ in prices_0]

        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")
        print(f"val = {valuations}\n budgets = {budgets}\n")

        # Linear Fisher Market
        if 'linear' in market_types:
            print(f"------ Linear Fisher Market ------")
            #TODO:market_typeはハードに指定しないとダメか？他の方法が無いか考える
            #TODO:time_averageと普通のやつを分けて指定できるようにする
            results["linear"].append(run_experiment(fm.calc_gda, get_obj_linear, 'linear', num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate_linear, mutation_rate[0], allocations_linear_ref, prices_linear_ref, num_iters, update_freq, arch))
            objective_value["linear"].append(minimize(get_obj_linear, prices_0, bounds=bounds, args=(allocations_0, budgets, valuations), method='SLSQP').fun)
            #print(objective_value["linear"][-1])
            #print(minimize(get_obj_linear, prices_0, bounds=bounds, args=(allocations_0, budgets, valuations), method='SLSQP').x)

        # Cobb-Douglas Fisher Market
        if 'cd' in market_types:
            print(f"------ Cobb-Douglas Fisher Market ------")
            #prices_0 = np.random.rand(num_goods)
            results["cd"].append(run_experiment(fm.calc_gda, get_obj_cd, 'cd', num_buyers, valuations_cd, budgets, allocations_0, prices_0, learning_rate_cd, mutation_rate[1], allocations_cd_ref, prices_cd_ref, num_iters, update_freq, arch))
            objective_value["cd"].append(minimize(get_obj_cd, prices_0, bounds=bounds, args=(allocations_0, budgets, valuations_cd), method='SLSQP').fun)
            #print(objective_value["cd"][-1])
            #print(minimize(get_obj_cd, prices_0, bounds=bounds, args=(allocations_0, budgets, valuations_cd), method='SLSQP').x)

        # Leontief Fisher Market
        if 'leontief' in market_types:
            print(f"------ Leontief Fisher Market ------")
            #prices_0 = np.random.rand(num_goods)
            results["leontief"].append(run_experiment(fm.calc_gda, get_obj_leontief, 'leontief', num_buyers, valuations, budgets, allocations_0, prices_0, learning_rate_leontief, mutation_rate[2], allocations_leontief_ref, prices_leontief_ref, num_iters, update_freq, arch))
            objective_value["leontief"].append(minimize(get_obj_leontief, prices_0, bounds=bounds, args=(allocations_0, budgets, valuations), method='SLSQP').fun)
            #print(objective_value["leontief"][-1])
            #print(minimize(get_obj_leontief, prices_0, bounds=bounds, args=(allocations_0, budgets, valuations), method='SLSQP').x)

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

def plot_and_save_obj_graphs_followed_paper(obj_hist_data, plot_titles, market_types, dir_obj, dir_graphs, arch):
        print("plotting exploitability graphs...")
        #fig, axs = plt.subplots(1, len(market_types), figsize=(25.5, 5.5))
        width_per_subplot = 8
        fig, axs = plt.subplots(1, len(market_types), figsize=(width_per_subplot * len(market_types), 5.5))

        if len(market_types) == 1:
            axs = [axs]

        for i, (obj_hist, title, market_type) in enumerate(zip(obj_hist_data, plot_titles, market_types)):
            mean_obj = np.mean(obj_hist, axis=0) - sum(objective_value[market_type]) / len(objective_value[market_type])
            #Goktasによると、下の式はt^{-1/2}で割るということらしい
            mean_obj = mean_obj.flatten()
            indices = np.arange(1, len(mean_obj)+1, 1)
            indices = indices ** (1/2)
            mean_obj = mean_obj * indices
            ### ここまで
            axs[i].plot(mean_obj, color="b")
            axs[i].set_title(title, fontsize="medium")
            axs[i].set_xlabel('Iteration Number', fontsize=21)
            axs[i].set_ylabel(r'Explotability$/t^{-1/2}$', fontsize=21)
            axs[i].grid(linestyle='dotted')
            #axs[i].set(xlabel='Iteration Number', ylabel=r'Explotability$/t^{-1/2}$', fontsize=22)
            #axs[i].set_ylim(-0.05, 3)
            pd.DataFrame(mean_obj).to_csv(f"{dir_obj}/{arch}_exploit_hist_{market_type}.csv")

        #fig.set_size_inches(25.5, 5.5)
        plt.rcParams["font.size"] = 22
        plt.subplots_adjust(wspace=0.4)
        plt.grid(linestyle='dotted')
        plt.savefig(f"{dir_graphs}/{arch}_exploit_graphs.pdf")
        plt.savefig(f"{dir_graphs}/{arch}_exploit_graphs.jpg")
        plt.close()

def plot_and_save_obj_graphs(obj_hist_data, plot_titles, market_types, dir_obj, dir_graphs, arch):
        print("plotting exploitability graphs...")
        #fig, axs = plt.subplots(1, len(market_types), figsize=(25.5, 5.5))
        width_per_subplot = 8
        fig, axs = plt.subplots(1, len(market_types), figsize=(width_per_subplot * len(market_types), 5.5))

        if len(market_types) == 1:
            axs = [axs]

        for i, (obj_hist, title, market_type) in enumerate(zip(obj_hist_data, plot_titles, market_types)):
            mean_obj = np.mean(obj_hist, axis=0) - np.min(np.mean(obj_hist, axis=0))
            #Goktasによると、下の式はt^{-1/2}で割るということらしい
            mean_obj = mean_obj.flatten()
            indices = np.arange(1, len(mean_obj)+1, 1)
            indices = indices ** (1/2)
            mean_obj = mean_obj * indices
            ### ここまで
            axs[i].plot(mean_obj, color="b")
            axs[i].set_title(title, fontsize="medium")
            axs[i].set_xlabel('Iteration Number', fontsize=21)
            axs[i].set_ylabel(r'Explotability$/t^{-1/2}$', fontsize=21)
            axs[i].grid(linestyle='dotted')
            #axs[i].set(xlabel='Iteration Number', ylabel=r'Explotability$/t^{-1/2}$', fontsize=22)
            #axs[i].set_ylim(-0.05, 3)
            pd.DataFrame(mean_obj).to_csv(f"{dir_obj}/{arch}_exploit_hist_{market_type}.csv")

        #fig.set_size_inches(25.5, 5.5)
        plt.rcParams["font.size"] = 22
        plt.subplots_adjust(wspace=0.4)
        plt.savefig(f"{dir_graphs}/{arch}_exploit_graphs.pdf")
        plt.savefig(f"{dir_graphs}/{arch}_exploit_graphs.jpg")
        plt.close()

def plot_and_save_prices_graphs(prices_hist_data, plot_titles, market_types, dir_prices, dir_graphs, arch):
    print("plotting prices graphs...")
    #fig, axs = plt.subplots(1, len(market_types))
    width_per_subplot = 8
    fig, axs = plt.subplots(1, len(market_types), figsize=(width_per_subplot * len(market_types), 5.5))

    if len(market_types) == 1:
        axs = [axs]
    
    for i, (prices_hist, title, market_type) in enumerate(zip(prices_hist_data, plot_titles, market_types)):
        mean_prices = np.mean(prices_hist, axis=0)
        axs[i].plot(mean_prices)
        axs[i].set_title(title, fontsize="medium")
        axs[i].set(xlabel='Iteration Number', ylabel=r'prices')
        axs[i].legend(prices_hist[0].columns)
        axs[i].grid(linestyle='dotted')
        pd.DataFrame(mean_prices).to_csv(f"{dir_prices}/{arch}_prices_hist_{market_type}_average.csv")

    #fig.set_size_inches(25.5, 5.5)
    plt.rcParams["font.size"] = 18
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(f"{dir_graphs}/{arch}_prices_graphs.jpg")
    plt.savefig(f"{dir_graphs}/{arch}_prices_graphs.pdf")
    plt.close()

def plot_and_save_allocations_graphs(plot_titles, market_types, dir_allocations, dir_graphs, arch, num_buyers):
    print("plotting allocations graphs...")
    fig, axs = plt.subplots()

    for market_type, plot_title in zip(market_types, plot_titles):
        for buyer in range(num_buyers):
            file_pattern = f"{dir_allocations}/{arch}_allocations_hist_{market_type}_*_buyer_{buyer}.csv"
            file_list = glob.glob(file_pattern)
            if not file_list:
                print("No files found.")
                return

            dfs = [pd.read_csv(file, index_col=0) for file in file_list]

            df_concat = pd.concat(dfs)
            df_mean = df_concat.groupby(df_concat.index).mean()
            pd.DataFrame(df_mean).to_csv(f"{dir_allocations}/{arch}_allocations_hist_{market_type}_buyer_{buyer}_average.csv")

            fig, axs = plt.subplots()
            df_mean.plot()
            fig.set_size_inches(25.5, 5.5)
            plt.title(plot_title+' buyer '+str(buyer), fontsize="medium")
            plt.xlabel('Iteration Number')
            plt.ylabel('Allocations')
            plt.rcParams["font.size"] = 18
            plt.subplots_adjust(wspace=0.4)
            plt.grid(linestyle='dotted')
            plt.savefig(f"{dir_graphs}/{arch}_allocations_graphs_{market_type}_buyer_{buyer}.pdf")
            plt.savefig(f"{dir_graphs}/{arch}_allocations_graphs_{market_type}_buyer_{buyer}.jpg")
            plt.close()

def get_dataframes(pattern, dir_content, dir_obj):
    files = [os.path.join(dir_obj, file) for file in dir_content if re.match(pattern, file)]
    return [pd.read_csv(file, index_col=0) for file in files]

def write_params_to_file(market_types, num_experiments, num_buyers, num_goods, learning_rate_linear, learning_rate_cd, learning_rate_leontief, mutation_rate, num_iters, update_freq, arch, dir_data):
    params = {
        "market_types": market_types,
        "num_experiments": num_experiments,
        "num_buyers": num_buyers,
        "num_goods": num_goods,
        "learning_rate_linear": learning_rate_linear,
        "learning_rate_cd": learning_rate_cd,
        "learning_rate_leontief": learning_rate_leontief,
        "mutation_rate": mutation_rate,
        "num_iters": num_iters,
        "update_freq": update_freq,
        "arch": arch,
    }

    with open(f'{dir_data}/args.json', 'w') as f:
        json.dump(params, f, indent=4)

def main(args):
    now = datetime.datetime.now()
    nowdate = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_data = Path(f'results/{nowdate}_{args.arch}_en_{args.num_experiments}_iters_{args.num_iters}_uf_{args.update_freq}')
    dir_obj = Path(f"{dir_data}/data/obj")
    dir_obj.mkdir(parents=True, exist_ok=True)
    dir_allocations = Path(f"{dir_data}/data/allocations")
    dir_allocations.mkdir(parents=True, exist_ok=True)
    dir_prices = Path(f"{dir_data}/data/prices")
    dir_prices.mkdir(parents=True, exist_ok=True)
    dir_graphs = Path(f"{dir_data}/graphs")
    dir_graphs.mkdir(parents=True, exist_ok=True)

    write_params_to_file(args.market_types, args.num_experiments, args.num_buyers, args.num_goods, args.learning_rate_linear, args.learning_rate_cd, args.learning_rate_leontief, args.mutation_rate, args.num_iters, args.update_freq, args.arch, dir_data)
    #TODO:収束先を改めて指定する
    #収束先
    allocations_linear_ref = np.array([[0.20623613091668674,0.3336534555336145,0.25645391534516004,0.10005288662062428,0.0980636054576395,0.15957711474141045,0.2309186253627186,0.18359554918334048],
    [0.1931747247023043,0.2195250487167748,0.16046510898485222,0.24722257213277707,0.29780321154228784,0.19393492406887672,0.12729652286708698,0.2553246241947231],
    [0.25202845564968596,0.11083088207187246,0.3557522541763148,0.11172855932178827,0.23114459259544257,0.3198853043815345,0.21456643185213134,0.10548584588588202],
    [0.10169290412476979,0.18458568138108583,0.09244625123933141,0.30978417183553836,0.1611064867693787,0.13422534819708098,0.25138380726970017,0.32612324446368196],
    [0.2451079619790657,0.16770222726333758,0.10757142420148716,0.24963582773178378,0.22454022404264057,0.16452348094116284,0.1518205626313237,0.1842317170172249]])
    prices_linear_ref = np.array([11.270709297039923,10.814289574822979,10.753916411533188,11.438802312898837,11.358676866398627,11.143766342313254,10.96435957422994,11.044599675122948])
    
    #収束先
    allocations_cd_ref = np.array([[0.19846068684605161,0.2041121907143629,0.20262525718257568,0.18312349691429142,0.1844024592114179,0.1873494193689325,0.20804282023694362,0.18224882587283434],
    [0.20479573444839408,0.20144794331693205,0.2223402113709073,0.2174516703271939,0.22466992275431963,0.20465091551153844,0.20084071473350865,0.21826001305096007],
    [0.1913010815311055,0.20441365681774631,0.20017584243442849,0.19130904171186264,0.20987884824120706,0.20828365977518543,0.20479247001381035,0.20301600650048401],
    [0.1983165183472086,0.20269793477684858,0.19452849683206833,0.21733792988747772,0.1851431558260249,0.2173201811158933,0.19901557872314724,0.2111616605231268],
    [0.2071259788272462,0.187328274374117,0.18033019218002716,0.19077786115918116,0.19590561396703773,0.18239582422845774,0.18730841629259698,0.1853134940526016]])
    prices_cd_ref = np.array([8.937202224509907,9.10385916965439,8.94938452915189,9.27881054935158,9.45135589367398,9.531523736566268,8.716504432102125,9.42891125502871])

    #収束先
    allocations_leontief_ref = np.array([[0.24121808127568772,0.24835034236601725,0.18681018364131857,0.1479421731841541,0.21432769253756412,0.18104435806645844,0.14267175236238952,0.1770879826287069],
    [0.09913842224810276,0.1563834057166346,0.153837723805674,0.28692497753669943,0.2659672550266936,0.23595130910842924,0.22219380979947861,0.2727379444891606],
    [0.23978549407714417,0.17959466465600807,0.2144583803155681,0.2112383263207295,0.13963709896731555,0.1689546414398923,0.166984852926709,0.28987072019123006],
    [0.22709575219822103,0.19082113465722358,0.19867309359149377,0.18581268518195992,0.15389701366534572,0.25116026316921264,0.2496535833175094,0.17056407915776964],
    [0.20091958961555217,0.17268207429998,0.25735653816104287,0.16715388619004495,0.17572702975094173,0.1644295661873398,0.2523986848045243,0.12512742828319517]])
    prices_leontief_ref = np.array([9.153076425449168,9.364909065859923,9.27838920430462,9.245566865329362,9.046316932360622,8.9900646175248,9.193457326837096,9.122314474779257])
    #prices_leontief_ref = np.array([11,11,11,11,11,11,11,11])
    #prices_leontief_ref = np.array([1.76612110e+01, 7.58249600e+00, 4.02669818e+00, 1.44980598e+01, 1.10000000e-05, 1.32743758e+01, 1.61819512e+01, 7.61927460e+00])

    # results
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
    #plot_and_save_obj_graphs(obj_hist_data, plot_titles, args.market_types, dir_obj, dir_graphs, args.arch)
    plot_and_save_obj_graphs_followed_paper(obj_hist_data, plot_titles, args.market_types, dir_obj, dir_graphs, args.arch)
    plot_and_save_prices_graphs(prices_hist_data, plot_titles, args.market_types, dir_prices, dir_graphs, args.arch)
    plot_and_save_allocations_graphs(plot_titles, args.market_types, dir_allocations, dir_graphs, args.arch, args.num_buyers)

if __name__ == '__main__':
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

    start = time.time()
    main(args)
    elapsed_time = time.time() - start
    
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")