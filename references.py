import numpy as np

class References:
  def __init__(self):
    self.allocations_linear_ref2 = np.array([[0.2140533516657123,0.3061049563869186,0.27166908707005966,0.09571924111289291,0.0901534275339205,0.1871724245988343,0.2388888704801567,0.17565218791642462],
    [0.19524636955165617,0.22399845066806826,0.1605925706417905,0.24230411054588,0.3008993418189746,0.16602469940876047,0.1674345879496748,0.23229037423201512],
    [0.2627728479780005,0.09143493544940536,0.32627628167202094,0.11954669073171782,0.21691127438984087,0.32869768533375454,0.23589615246925652,0.10651353133012238],
    [0.10539172895354212,0.1961667008995463,0.09940269338520986,0.2804478412546389,0.17582705778263175,0.1666113991243431,0.26783814219447943,0.29808539069440976],
    [0.2520448172229872,0.1782818927609587,0.10388683374179257,0.24669910164714598,0.23240726975797948,0.17174820153877457,0.1596453960270177,0.1714572420480117]])
    self.prices_linear_ref2 = np.array([10.305009236665587,9.867552078184502,9.765631740425116,10.503610801252867,10.484207649086361,10.155319357469878,10.06851615467792,10.114892830529984])
    self.allocations_linear_ref = np.full_like(self.allocations_linear_ref2, 0.1)
    self.prices_linear_ref = np.full_like(self.prices_linear_ref2, 5)

    self.allocations_cd_ref2 = np.array([[0.19882207346959496,0.20454969041013799,0.20301853791617144,0.18350679664792788,0.1848263104906626,0.18776534860985633,0.208349163569394,0.18270868999562612],
    [0.20511539324555583,0.20183085431590692,0.2227389942634766,0.21795332966637165,0.22514749605715467,0.20514626859681692,0.20115062832681754,0.21874227275737973],
    [0.1916512561780199,0.20479231693757433,0.20055768285824555,0.1917134249899315,0.21031217155416812,0.20873926276662164,0.20508245039446735,0.20350010150683548],
    [0.19867563522246953,0.20308858101617866,0.19485206609760053,0.21784490422507607,0.18558821395318137,0.21780528060932497,0.19931167738756894,0.21167614148273586],
    [0.20747429147055482,0.1876836062006191,0.1806401069787935,0.19117202699863203,0.1963349739329846,0.18279746454826035,0.18756410239004,0.18577336361679564]])
    self.prices_cd_ref2 = np.array([8.919804195898308,9.083935664624336,8.931057250091445,9.255373557174382,9.428494518784056,9.508329733039915,8.702703697870284,9.402731490182411])
    self.allocations_cd_ref = np.full_like(self.allocations_cd_ref2, 0.1)
    self.prices_cd_ref = np.full_like(self.prices_cd_ref2, 5)

    self.allocations_leontief_ref2 = np.array([[0.12866456874290486,0.18544965057375923,0.15668769709597827,0.2788845306987129,0.26668936860283254,0.12997920374110544,0.21050759357411178,0.19361244009502113],
    [0.1681461228769793,0.24378126805301706,0.21749649338531438,0.14355888559827204,0.2187185913728465,0.21081826828905736,0.19868437404216674,0.29914133559530093],
    [0.2727733867992524,0.20427932832241838,0.20619830345926266,0.2869555750062897,0.14732658456364273,0.19226849018590775,0.1340008015079105,0.17044016955142247],
    [0.23462187856827854,0.1636721017430995,0.21981513033975159,0.1560446702010896,0.2246448731638046,0.20863863957370196,0.2599558057820761,0.16525922115090244],
    [0.17115167200961734,0.1913316969652631,0.19916199827895448,0.13210022527618612,0.15419022886931527,0.24289971587504913,0.2115231787158105,0.2133065056036007]])
    self.prices_leontief_ref2 = np.array([9.160170331214566,9.158025644512737,9.159421329401761,9.159060411237837,9.157849830207363,9.158670796130883,9.158345305893103,9.158696747562974])
    self.allocations_leontief_ref = np.full_like(self.allocations_leontief_ref2, 0.1)
    self.prices_leontief_ref = np.full_like(self.prices_leontief_ref2, 5)