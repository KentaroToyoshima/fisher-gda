import os
import re
import json
import pandas as pd
import datetime
from pathlib import Path

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

def create_directories(arch, num_experiments, num_iters, update_freq):
    now = datetime.datetime.now()
    nowdate = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_data = Path(f'results/{nowdate}_{arch}_en_{num_experiments}_iters_{num_iters}_uf_{update_freq}')
    dir_obj = Path(f"{dir_data}/data/obj")
    dir_obj.mkdir(parents=True, exist_ok=True)
    dir_allocations = Path(f"{dir_data}/data/allocations")
    dir_allocations.mkdir(parents=True, exist_ok=True)
    dir_prices = Path(f"{dir_data}/data/prices")
    dir_prices.mkdir(parents=True, exist_ok=True)
    dir_graphs = Path(f"{dir_data}/graphs")
    dir_graphs.mkdir(parents=True, exist_ok=True)
    return dir_data, dir_obj, dir_allocations, dir_prices, dir_graphs