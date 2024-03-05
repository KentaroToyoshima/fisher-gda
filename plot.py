from operator import index
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy.optimize import minimize

def plot_and_save_obj_graphs(obj_hist_data, plot_titles, market_types, dir_obj, dir_graphs, arch, objective_value):
        print("plotting exploitability graphs...")
        #fig, axs = plt.subplots(1, len(market_types), figsize=(25.5, 5.5))
        width_per_subplot = 8
        fig, axs = plt.subplots(1, len(market_types), figsize=(width_per_subplot * len(market_types), 5.5))

        if len(market_types) == 1:
            axs = [axs]

        for i, (obj_hist, title, market_type) in enumerate(zip(obj_hist_data, plot_titles, market_types)):
            mean_obj = np.mean(obj_hist, axis=0) - sum(objective_value[market_type]) / len(objective_value[market_type])
            axs[i].plot(mean_obj, color="b")
            axs[i].set_title(title, fontsize=22)
            axs[i].set_xlabel('Iteration Number', fontsize=21)
            axs[i].set_ylabel(r'Exploitability', fontsize=21)
            axs[i].grid(linestyle='dotted')
            pd.DataFrame(mean_obj).to_csv(f"{dir_obj}/{arch}_exploit_hist_{market_type}.csv")

        #fig.set_size_inches(25.5, 5.5)
        plt.rcParams["font.size"] = 22
        plt.subplots_adjust(wspace=0.4)
        plt.grid(linestyle='dotted')
        plt.savefig(f"{dir_graphs}/{arch}_exploit_graphs.pdf")
        plt.savefig(f"{dir_graphs}/{arch}_exploit_graphs.jpg")
        plt.close()

'''
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
            #mean_obj = mean_obj.flatten()
            #indices = np.arange(1, len(mean_obj)+1, 1)
            #indices = indices ** (1/2)
            #mean_obj = mean_obj * indices
            ### ここまで
            axs[i].plot(mean_obj, color="b")
            axs[i].set_title(title, fontsize="medium")
            axs[i].set_xlabel('Iteration Number', fontsize=21)
            axs[i].set_ylabel(r'Exploitability$/t^{-1/2}$', fontsize=21)
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
'''

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
        axs[i].set_xlabel('Iteration Number', fontsize=21)
        axs[i].set_ylabel('prices', fontsize=21)
        axs[i].legend(prices_hist[0].columns)
        axs[i].grid(linestyle='dotted')
        pd.DataFrame(mean_prices).to_csv(f"{dir_prices}/{arch}_prices_hist_{market_type}_average.csv")

    #fig.set_size_inches(25.5, 5.5)
    plt.rcParams["font.size"] = 18
    plt.subplots_adjust(wspace=0.4, bottom=0.15)
    plt.savefig(f"{dir_graphs}/{arch}_prices_graphs.jpg")
    plt.savefig(f"{dir_graphs}/{arch}_prices_graphs.pdf")
    plt.close()

def plot_and_save_allocations_graphs(plot_titles, market_types, dir_allocations, dir_graphs, arch, num_buyers):
    print("plotting allocations graphs...")

    n_cols = round(np.sqrt(num_buyers))
    n_rows = round(np.sqrt(num_buyers))

    if n_cols * n_rows < num_buyers:
        n_cols += 1

    for market_type, plot_title in zip(market_types, plot_titles):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 9*n_rows), squeeze=False)

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

            # Calculate subplot row and column indices
            row_idx = buyer // n_cols
            col_idx = buyer % n_cols

            df_mean.plot(ax=axs[row_idx, col_idx])  # Specify the axis to plot
            axs[row_idx, col_idx].set_title(plot_title+' buyer '+str(buyer), fontsize="medium")
            axs[row_idx, col_idx].set_xlabel('Iteration Number')
            axs[row_idx, col_idx].set_ylabel('Allocations')
            axs[row_idx, col_idx].grid(linestyle='dotted')

        # Remove empty subplots
        for idx in range(num_buyers, n_cols*n_rows):
            fig.delaxes(axs.flatten()[idx])

        plt.tight_layout()
        plt.rcParams["font.size"] = 18
        plt.savefig(f"{dir_graphs}/{arch}_allocations_graphs_{market_type}_all_buyers.pdf")
        plt.savefig(f"{dir_graphs}/{arch}_allocations_graphs_{market_type}_all_buyers.jpg")
        plt.close()