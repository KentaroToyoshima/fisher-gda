from scipy.optimize import minimize
import numpy as np

def get_linear_indirect_utill(prices, budget, valuations):
    return np.max(valuations/prices)*budget

# Objective Functions for linear
def get_obj_linear(prices, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = get_linear_indirect_utill(prices.clip(min= 0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min = 0.01))

if __name__ == '__main__':
    # 初期値
    num_buyers = 5
    num_goods = 8
    valuations = np.random.rand(num_buyers, num_goods) * 10 + 5
    budgets = np.random.rand(num_buyers) * 10 + 10
    prices_0 = np.random.rand(num_goods)

    # 現在の反復回数
    iteration_count = 0

    # 動的に変化する不等式制約条件
    def constraint_function(x):
        global iteration_count
        iteration_count += 1
        return x[0] + x[1] - 1.0 - 0.01 * iteration_count

    # 等式制約条件 (例: x[0] + x[1] = 1.5)
    #constraints = [{'type': 'eq', 'fun': lambda x:  x[0] + x[1] - 1.5}]
    #constraints = [{'type': 'ineq', 'fun': constraint_function}]

    # 最適化の実行
    result = minimize(get_obj_linear, prices_0, args=(budgets, valuations), method='SLSQP')

    print("Optimized parameters:", result.x)  # 最適解
    print("Objective function value at optimized parameters:", result.fun)  # 最適化されたパラメータでの目的関数の値
    
