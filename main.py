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

# Objective Functions for linear, Cobb-Douglas, and Leontief
def get_obj_linear(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_linear_indirect_utill(prices.clip(min= 0.0001), b, v)
        #utils[buyer] = cu.get_linear_utility(demands[buyer,:], v)
    # return np.sum(prices) + budgets.T @ np.log(utils) + np.sum(budgets) - np.sum(demands @ prices )
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.01))

def get_obj_cd(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_CD_indirect_util(prices.clip(min= 0.0001), b, v)
        #utils[buyer] = cu.get_CD_utility(demands[buyer,:], v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001)) 

def get_obj_leontief(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_leontief_indirect_util(prices.clip(min= 0.0001), b, v)
        #utils[buyer] = cu.get_leontief_utility(demands[buyer,:], v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001))

# Function that run Max-Oracle Gradient Descent and Nested Gradient Descent Ascent Tests and returns data
def run_experiment(fm_func, get_obj, market_type, num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, update_freq, arch):
    demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm_func(num_buyers, valuations, budgets, demands_0, prices_0, learning_rate, mutation_rate, demands_ref, prices_ref, num_iters, update_freq, arch, market_type)
    objective_values = [get_obj(p, x, budgets, valuations) for p, x in zip(prices_hist_gda, demands_hist_gda)]
    return demands_hist_gda, prices_hist_gda, objective_values

def run_test(num_buyers, num_goods, demands_linear_ref, demands_cd_ref, demands_leontief_ref, prices_linear_ref, prices_cd_ref, prices_leontief_ref, learning_rate_linear, learning_rate_cd, learning_rate_leontief, mutation_rate, num_experiments, num_iters, update_freq, arch, market_types, dir_obj, dir_demands, dir_prices):
    results = {key: [] for key in market_types}

    for experiment_num in range(num_experiments):
        np.random.seed(experiment_num)
        random.seed(experiment_num)
        valuations = np.random.rand(num_buyers, num_goods) * 10 + 5
        valuations_cd = (valuations.T / np.sum(valuations, axis=1)).T
        budgets = np.random.rand(num_buyers) * 10 + 10
        #demands_0 = np.zeros(valuations.shape)
        demands_0 = np.random.rand(num_buyers, num_goods)
        prices_0 = np.random.rand(num_goods) * 5

        print(f"************* Experiment: {experiment_num + 1}/{num_experiments} *************")
        print(f"****** Market Parameters ******\nval = {valuations}\n budgets = {budgets}\n")
        print(f"*******************************")
        print(f"------------ GDA ------------")

        # Linear Fisher Market
        print(f"------ Linear Fisher Market ------")
        results["linear"].append(run_experiment(fm.calc_gda, get_obj_linear, market_types[0], num_buyers, valuations, budgets, demands_0, prices_0, learning_rate_linear, mutation_rate[0], demands_linear_ref, prices_linear_ref, num_iters, update_freq, arch))

        # Cobb-Douglas Fisher Market
        print(f"------ Cobb-Douglas Fisher Market ------")
        #prices_0 = np.random.rand(num_goods)
        results["cd"].append(run_experiment(fm.calc_gda, get_obj_cd, market_types[1], num_buyers, valuations_cd, budgets, demands_0, prices_0, learning_rate_cd, mutation_rate[1], demands_cd_ref, prices_cd_ref, num_iters, update_freq, arch))

        # Leontief Fisher Market
        print(f"------ Leontief Fisher Market ------")
        #prices_0 = np.random.rand(num_goods)
        results["leontief"].append(run_experiment(fm.calc_gda, get_obj_leontief, market_types[2], num_buyers, valuations, budgets, demands_0, prices_0, learning_rate_leontief, mutation_rate[2], demands_leontief_ref, prices_leontief_ref, num_iters, update_freq, arch))

        # Save data
        for key in results:
            demands_hist, prices_hist, obj_hist = zip(*results[key])

            demands_hist_all = np.array(demands_hist[-1]).swapaxes(0, 1)
            prices_hist_all = np.array(prices_hist[-1])
            obj_hist_all = np.array(obj_hist[-1])

            # Save prices
            df_prices = pd.DataFrame(prices_hist_all)
            df_prices.to_csv(f"{dir_prices}/{arch}_prices_hist_gda_{key}_{experiment_num}.csv")

            # Save demands
            for buyer in range(num_buyers):
                df_demands = pd.DataFrame(demands_hist_all[buyer])
                df_demands.to_csv(f"{dir_demands}/{arch}_demands_hist_gda_{key}_{experiment_num}_buyer_{buyer}.csv")

            # Save objective function
            df_obj = pd.DataFrame(obj_hist_all)
            df_obj.to_csv(f"{dir_obj}/{arch}_obj_hist_gda_{key}_{experiment_num}.csv")

    #TODO:返り値が必要か考える
    return (
        results["linear"],
        results["cd"],
        results["leontief"],
    )

def plot_and_save_obj_graphs(obj_hist_data, plot_titles, file_prefix, dir_obj, dir_graphs, arch):
        fig, axs = plt.subplots(1, 3)
    
        for i, (obj_hist, title, market) in enumerate(zip(obj_hist_data, plot_titles, file_prefix)):
            mean_obj = np.mean(obj_hist, axis=0) - np.min(np.mean(obj_hist, axis=0))
            axs[i].plot(mean_obj, color="b")
            axs[i].set_title(title, fontsize="medium")
            axs[i].set(xlabel='Iteration Number', ylabel=r'Explotability')
            #axs[i].set_ylim(-0.05, 3)
            pd.DataFrame(mean_obj).to_csv(f"{dir_obj}/{arch}_exploit_hist_{market}.csv")

        fig.set_size_inches(18.5, 5.5)
        plt.rcParams["font.size"] = 18
        plt.subplots_adjust(wspace=0.4)
        plt.savefig(f"{dir_graphs}/{arch}_obj_graphs.jpg")

def create_average_file(market_type, buyer_num, dir_path, arch):
    # 同じfunction_typeとbuyer_numを持つすべてのCSVファイルを見つける
    file_pattern = f"{dir_path}/{arch}_demands_hist_{market_type}_*_buyer_{buyer_num}.csv"
    file_list = glob.glob(file_pattern)

    if not file_list:
        print("No files found.")
        return

    # 各ファイルからデータフレームを読み込み、リストに追加する
    dfs = [pd.read_csv(file) for file in file_list]

    # すべてのデータフレームを結合し、列ごとに平均を計算する
    df_concat = pd.concat(dfs)
    df_mean = df_concat.groupby(df_concat.index).mean()

    # 平均データを新しいCSVファイルに書き出す
    output_file = f"{dir_path}/{arch}_demands_hist_{market_type}_average_buyer_{buyer_num}.csv"
    df_mean.to_csv(output_file, index=False)

    print(f"Average file created: {output_file}")

# TODO:demandの推移をプロット
def plot_and_save_demand_graphs(demands_hist_data, plot_titles, file_prefix, dir_demands, dir_graphs, arch, num_buyers):
    pass

def plot_and_save_prices_graphs(prices_hist_data, plot_titles, file_prefix, dir_prices, dir_graphs, arch):
    fig, axs = plt.subplots(1, 3)
    
    for i, (prices_hist, title, market) in enumerate(zip(prices_hist_data, plot_titles, file_prefix)):
        mean_prices = np.mean(prices_hist, axis=0)
        axs[i].plot(mean_prices)
        axs[i].set_title(title, fontsize="medium")
        axs[i].set(xlabel='Iteration Number', ylabel=r'prices')
        #axs[i].set_ylim(-0.05, 3)
        pd.DataFrame(mean_prices).to_csv(f"{dir_prices}/{arch}_prices_hist_{market}.csv")

    fig.set_size_inches(25.5, 5.5)
    plt.rcParams["font.size"] = 18
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(f"{dir_graphs}/{arch}_prices_graphs.jpg")
    plt.savefig(f"{dir_graphs}/{arch}_prices_graphs.pdf")

"""
def plot_and_save_prices_graphs(prices_hist_data, plot_titles, file_prefix, dir_prices, dir_graphs, arch):
    fig, axs = plt.subplots(1, 3)
    
    # matplotlibの色サイクルを取得
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (prices_hist, title, market) in enumerate(zip(prices_hist_data, plot_titles, file_prefix)):
        mean_prices = np.mean(prices_hist, axis=0)
        # 色サイクルから色を選択
        axs[i].plot(mean_prices, color=colors[i % len(colors)])
        axs[i].set_title(title, fontsize="medium")
        axs[i].set(xlabel='Iteration Number', ylabel=r'prices')
        #axs[i].set_ylim(-0.05, 3)
        pd.DataFrame(mean_prices).to_csv(f"{dir_prices}/{arch}_prices_hist_{market}.csv")

    fig.set_size_inches(18.5, 5.5)
    plt.rcParams["font.size"] = 18
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(f"{dir_graphs}/{arch}_prices_graphs.jpg")
    plt.show()
"""

def get_dataframes(pattern, dir_content, dir_obj):
    files = [os.path.join(dir_obj, file) for file in dir_content if re.match(pattern, file)]
    return [pd.read_csv(file, index_col=0) for file in files]

if __name__ == '__main__':
    market_types = ['linear', 'cd', 'leontief']
    num_experiments = 5
    num_buyers = 5
    num_goods = 8
    learning_rate_linear =  [2, 0.1]  #[price_lr, demand_lr]
    learning_rate_cd = [2, 0.1]
    learning_rate_leontief = [2, 0.1]
    mutation_rate = [1, 1, 1] #[linear, cd, leon]
    num_iters= 1000
    update_freq = 0
    arch = 'alg4'

    now = datetime.datetime.now()
    nowdate = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_data = Path(f"results/{nowdate}_{arch}_en_{num_experiments}_iters_{num_iters}_ml_{mutation_rate}_uf_{update_freq}")
    dir_obj = Path(f"{dir_data}/data/obj")
    dir_obj.mkdir(parents=True, exist_ok=True)
    dir_demands = Path(f"{dir_data}/data/demands")
    dir_demands.mkdir(parents=True, exist_ok=True)
    dir_prices = Path(f"{dir_data}/data/prices")
    dir_prices.mkdir(parents=True, exist_ok=True)
    dir_graphs = Path(f"{dir_data}/graphs")
    dir_graphs.mkdir(parents=True, exist_ok=True)

    #収束先
    '''
    demands_linear_ref = np.array([[0.15870897253843697,0.18667388393005538,0.18262319641795074,0.23823041197502504,0.216446175522833,0.18785419361586111,0.23285530284893508,0.20507746522566062],
[0.2214629722042028,0.20860808797843736,0.22140787720885277,0.12582867717934845,0.21301308255590082,0.17276529955610048,0.24522152090120686,0.18522372158327838],
[0.20598304467483747,0.18441735271750875,0.177394288510474,0.2572936427742071,0.17507383048299144,0.17740357143142582,0.2578786629494514,0.17874099018154493],
[0.21133044246662247,0.23164742744370292,0.16653482755924817,0.21893520535502375,0.23551592001901875,0.22959460407602592,0.12318154087284972,0.20130681762206293],
[0.19931382866220138,0.18941322073160685,0.2480220856128333,0.16454023899668344,0.16201429058635855,0.23105622770410242,0.1434239927283041,0.2317231818367359]])
    prices_linear_ref = np.array([11.561, 11.72,  11.67,  11.247, 11.496, 11.587, 11.433, 11.412])
    '''
    demands_linear_ref = np.array([[0.0402328804194158,0.3869517814919724,0.31275162567579107,0.10044924702250246,0.013483744814439957,0.15126264008536944,0.2745819237139086,0.18021942972789157],
[0.19699406979479356,0.22071433982361538,0.19889282007752254,0.09999999999999996,0.24309699458679218,0.36222719437139034,0.10247265960832568,0.11069718614658564],
[0.2909620262450237,0.09605503675356876,0.06791190861246289,0.5811393171069016,0.3009579627122072,0.09417107607136978,0.10106381855632159,0.24442477072270022],
[0.3631587219062314,0.2232972168039405,0.1338548484431208,0.0,0.046240931882601755,0.1642187228049048,0.3055206954927026,0.30126829116521586],
[0.12129726016601235,0.07878879479173947,0.30017126713227027,0.24766918631790005,0.40808974935594106,0.21325956897499648,0.20374534679357895,0.15727645853439082]])
    prices_linear_ref = np.array([9.781, 10.811, 10.775, 10.799, 11.359, 11.316, 11.157, 11.19])
    
    #収束先
    '''
    demands_cd_ref = np.array([[0.1995556914054147,0.20165524477521304,0.19931980223021523,0.21632457826934778,0.20787097054079648,0.20039455450426125,0.2105604994073406,0.1938437875510947],
[0.2041706251959548,0.2026957637286098,0.2101141903995081,0.18036968807710924,0.20207163553748636,0.19497829771435835,0.20384670607543312,0.19860233725220827],
[0.20161655204518536,0.20518956524621643,0.19713009601735923,0.21690380709960821,0.1974723146980503,0.2106600686225747,0.20725089619422657,0.20592135061060687],
[0.19514304080621675,0.2000556540932039,0.19788390292250566,0.19621578711161786,0.19972305547030714,0.19996478878803362,0.19595274477178917,0.2027797062542929],
[0.19951409054722855,0.19040377215675672,0.19555200843041173,0.19018613944231672,0.19286202375335956,0.19400229037077227,0.18238915355121046,0.19885281833179747]])
    prices_cd_ref = np.array([9.412, 9.289, 9.58,  9.127, 9.659, 9.425, 9.598, 9.434])
    '''
    demands_cd_ref = np.array([[0.16785585176886186,0.22280678842880106,0.1933076800055864,0.1882063864850191,0.16804602786622735,0.1747180470680536,0.2152848745230623,0.1938129059859049],
[0.2067285655928333,0.18577363582972822,0.20781837442650128,0.17335676551701074,0.20108807970237613,0.20583337435582671,0.1656106247403229,0.1913652055659117],
[0.20566089134830104,0.1894173648479463,0.18481312035128555,0.26991915749204426,0.2211148857996958,0.23403395002917637,0.20615421312375565,0.2114903689967186],
[0.2298696937454078,0.18947403716621497,0.21763279454985912,0.17038365724263008,0.19730761407807684,0.1990749607589616,0.2014147539477415,0.1952369408163535],
[0.189884997544596,0.21252817372730962,0.19642803066676773,0.19813403326329582,0.21244339255362385,0.18633966778798175,0.21153553366511751,0.20809457863511138]])
    prices_cd_ref = np.array([8.35,  8.715, 8.897, 8.251, 9.818, 9.394,9.074, 9.521])

    #収束先
    '''
    demands_leontief_ref = np.array([[0.19705567341885252,0.1982915575355722,0.19773807989732078,0.21409476185120305,0.20372537548710448,0.2010255929828345,0.21065061187174436,0.1905887832707092],
[0.1992753959495068,0.19548523136325677,0.20307522600549718,0.17542932866144834,0.19647843442865903,0.18795989989276596,0.19501276061483022,0.1915799048637715],
[0.19424500982922424,0.19621123887398068,0.18764810592723333,0.2024828158026641,0.18337615275460595,0.19881062937452743,0.19567830874069533,0.19343444099148158],
[0.19383242365257075,0.19887022478796612,0.19875305914565522,0.19467848421904685,0.1976601041748803,0.1985466548518853,0.19833681676842804,0.20356241608579181],
[0.19435387768807794,0.1859283143415589,0.19023001142886942,0.18792336030600126,0.18723355777594416,0.18994284934439837,0.1782275204450602,0.19480714304994362]])
    prices_leontief_ref = np.array([10.55,   8.292, 10.686, 10.273,  6.996,  9.43,  11.713,  7.312])
    '''
    
    demands_leontief_ref = np.array([[0.16690964247024526,0.21846739892633,0.1909901239144872,0.1818970280103424,0.1722902884050845,0.1717841543205905,0.21213255665790656,0.1879734022234504],
[0.2106384804413337,0.1838506538152029,0.20969150584732804,0.17247546401269034,0.19741985216655406,0.2079426225077476,0.16617644016662073,0.19594067854381056],
[0.1948824360486758,0.1797189540387727,0.1808319392906698,0.24399369161730894,0.22020690189796593,0.22460357521355237,0.19029420826303525,0.20743288879799202],
[0.21638747714564688,0.1876567815742717,0.20953453116154255,0.16745256852712004,0.1940693670455123,0.19830574628829017,0.2001677775564986,0.19486708364026237],
[0.18133120346429715,0.20178200432474985,0.18675927048866053,0.1902912683470696,0.20276200368905756,0.18157319210162318,0.2003162265802289,0.20115939534650734]])
    prices_leontief_ref = np.array([7.376,  8.275,  7.249, 3.733, 11.582, 15.649, 14.015,  4.139])
    #prices_leontief_ref = np.array([11,11,11,11,11,11,11,11])
    #prices_leontief_ref = np.array([1.76612110e+01, 7.58249600e+00, 4.02669818e+00, 1.44980598e+01, 1.10000000e-05, 1.32743758e+01, 1.61819512e+01, 7.61927460e+00])
    '''
    demands_leontief_ref = np.array([[0.19390208048911095,0.19220917864873335,0.19656403260620897,0.19648171753194218,0.17918576274294395,0.17793491708795114,0.20984471787012993,0.1882996419795377],
[0.1931032192617211,0.19370810888302642,0.18288167646040607,0.2012558481639905,0.173924217043141,0.18615263610945867,0.20477942805795366,0.2157558221166424],
[0.1930468045928914,0.22455340470574123,0.20251390899102928,0.18746489328774532,0.2053368355349651,0.23555729722538182,0.2208308563353972,0.18349360660923372],
[0.21347019696134742,0.19490987325647438,0.2009186406840245,0.20362795836681102,0.23218015434880593,0.19772786953288735,0.18396847404534875,0.1910209181182973],
[0.1783267705183705,0.1769914154538792,0.18792892701302616,0.185286385399492,0.18266342409254785,0.16789708065984046,0.15434401156449973,0.20079940427245221]])
    prices_leontief_ref = np.array([1e-05,1e-05,1e-05,1e-05,1e-05,1e-05,1e-05,87.71031009096859])
    prices_leontief_ref = np.array([1e-05,11.356530129627775,1e-05,9.265022574604739,1e-05,1e-05,52.285013075347244,1e-05])
    '''

    # results
    (results_linear, results_cd, results_leontief) = run_test(num_buyers, num_goods, demands_linear_ref, demands_cd_ref, demands_leontief_ref, prices_linear_ref, prices_cd_ref, prices_leontief_ref, learning_rate_linear, learning_rate_cd, learning_rate_leontief, mutation_rate, num_experiments, num_iters, update_freq, arch, market_types, dir_obj, dir_demands, dir_prices)

    patterns = [rf'.*{key}.*\.csv' for key in market_types]
    dir_content = os.listdir(dir_obj)
    obj_hist_data = [get_dataframes(pattern, dir_content, dir_obj) for pattern in patterns]
    dir_content = os.listdir(dir_prices)
    prices_hist_data = [get_dataframes(pattern, dir_content, dir_prices) for pattern in patterns]
    dir_content = os.listdir(dir_prices)
    demands_hist_data = [get_dataframes(pattern, dir_content, dir_demands) for pattern in patterns]
    plot_titles = ["Linear Market", "Cobb-Douglas Market", "Leontief Market"]
    file_prefix = ["gda_linear", "gda_cd", "gda_leontief"]

    plot_and_save_obj_graphs(obj_hist_data, plot_titles, file_prefix, dir_obj, dir_graphs, arch)
    plot_and_save_prices_graphs(prices_hist_data, plot_titles, file_prefix, dir_prices, dir_graphs, arch)
    plot_and_save_demand_graphs(prices_hist_data, plot_titles, file_prefix, dir_demands, dir_graphs, arch, num_buyers)