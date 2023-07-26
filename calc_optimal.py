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
    
    # prices_histのデータ
    prices_hist = [
        np.array([0.265, 0.523, 0.094, 0.576, 0.929, 0.319, 0.667, 0.132]), 
        np.array([0.449, 0.719, 0.283, 0.769, 1.107, 0.514, 0.848, 0.335]), 
        np.array([0.667, 0.947, 0.505, 0.995, 1.318, 0.742, 1.063, 0.571]), 
        np.array([0.886, 1.177, 0.73 , 1.223, 1.531, 0.971, 1.279, 0.808]), 
        np.array([1.106, 1.407, 0.954, 1.451, 1.744, 1.201, 1.495, 1.045]), 
        np.array([1.325, 1.636, 1.178, 1.677, 1.957, 1.429, 1.711, 1.281]), 
        np.array([1.544, 1.864, 1.402, 1.903, 2.169, 1.657, 1.927, 1.516]), 
        np.array([1.761, 2.09 , 1.623, 2.127, 2.379, 1.882, 2.141, 1.749]), 
        np.array([1.977, 2.314, 1.844, 2.349, 2.588, 2.106, 2.353, 1.979]), 
        np.array([2.19 , 2.536, 2.062, 2.569, 2.795, 2.327, 2.564, 2.207])
    ]
    
    # NumPy配列に変換
    prices_hist_array = np.array(prices_hist)
    
    # 平均価格ベクトルを格納するためのリストを作成
    average_price_list = []
    
    # 各時点での価格ベクトルを、t=0からt=tまでの平均として保存
    for i in range(1, len(prices_hist_array) + 1):
        average_price = np.mean(prices_hist_array[:i], axis=0)
        average_price_list.append(average_price)
    
    for idx, avg_price in enumerate(average_price_list, start=1):
        print(f"Average price up to t={idx}: {avg_price}")
