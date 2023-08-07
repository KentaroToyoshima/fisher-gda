import os
import re
import json
import pandas as pd

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