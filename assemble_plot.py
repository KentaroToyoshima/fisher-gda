import pandas as pd
import matplotlib.pyplot as plt

# ファイルパスのリスト
m_alg2_files = [
    "results/2023_09_06_12_37_49_341953_m-alg2_en_20_iters_10000_uf_500/data/obj/m-alg2_exploit_hist_linear.csv",
    #"results/2023_09_13_11_14_29_060526_m-alg2_en_5_iters_10000_uf_20/data/obj/m-alg2_exploit_hist_linear.csv",
    "results/2023_09_06_12_37_49_341953_m-alg2_en_20_iters_10000_uf_500/data/obj/m-alg2_exploit_hist_cd.csv",
    #"results/2023_09_13_11_04_27_943169_m-alg2_en_5_iters_10000_uf_20/data/obj/m-alg2_exploit_hist_cd.csv",
    "results/2023_09_06_12_37_49_341953_m-alg2_en_20_iters_10000_uf_500/data/obj/m-alg2_exploit_hist_leontief.csv"
    #"results/2023_09_13_11_15_57_459557_m-alg2_en_5_iters_10000_uf_20/data/obj/m-alg2_exploit_hist_leontief.csv"
]

alg2_files = [
    "results/2023_09_06_12_30_36_063026_alg2_en_20_iters_10000_uf_500/data/obj/alg2_exploit_hist_linear.csv",
    "results/2023_09_06_12_30_36_063026_alg2_en_20_iters_10000_uf_500/data/obj/alg2_exploit_hist_cd.csv",
    "results/2023_09_06_12_30_36_063026_alg2_en_20_iters_10000_uf_500/data/obj/alg2_exploit_hist_leontief.csv"
]

labels = ['Linear Market', 'Cobb-Douglas Market', 'Leontief Market']
fontsize=18
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 各ファイルを読み込み、サブプロットに描画
for i in range(3):
    # LGDA (alg2) を青で描画
    data_alg2 = pd.read_csv(alg2_files[i], index_col=0)
    axs[i].plot(data_alg2, label='LGDA', color='blue')
    
    # M-LGDA (m-alg2) を赤で描画
    data_m_alg2 = pd.read_csv(m_alg2_files[i], index_col=0)
    axs[i].plot(data_m_alg2, label='LGDA with Mutation', color='red')
    
    axs[i].set_title(labels[i], fontsize=fontsize)
    axs[i].set_xlabel('Iterations', fontsize=fontsize)
    axs[i].set_ylabel('Exploitability', fontsize=fontsize)
    axs[i].grid(linestyle='dotted')
    #axs[i].set_xscale('log')
    #axs[i].set_ylim(-0.1, 80)  # y軸の最大値を50に制限
    #axs[i].set_yscale('log')
    if i==2:
        axs[i].set_ylim(-0.1, 60)  # y軸の最大値を50に制限
    axs[i].legend()
plt.tight_layout()
plt.savefig('comparison_graph.pdf')
plt.show()
