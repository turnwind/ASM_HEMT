from bayes_opt import BayesianOptimization
import time
import json
import DC_modeling
import DataExtraction
from DC_modeling import Getdatas
import numpy as np

mds_input = DataExtraction.loadmeasures(12)
mds_trans_lin = DataExtraction.loadmeasures(7)
mds_trans_sub = DataExtraction.loadmeasures(10)
mds_trans = DataExtraction.loadmeasures(9)
mds_out = DataExtraction.loadmeasures(8)



# 定义目标函数
def objective_function(voff,nfactor,u0,ua,Igsdio,Njgs,Igddio,Njgd,Rshg,Eta0,Vdscale,Cdscd,Rsc,Rdc,UTE,RTH0,LAMBDA,Vsat,Tbar):
    paras = {
        "voff": voff, "nfactor": nfactor, "u0": u0, "ua": ua, "Igsdio": Igsdio, "Njgs": Njgs,
        "Igddio": Igddio, "Njgd": Njgd, "Rshg": Rshg, "Eta0": Eta0, "Vdscale": Vdscale,
        "Cdscd": Cdscd, "Rsc": Rsc, "Rdc": Rdc, "UTE": UTE, "RTH0": RTH0, "LAMBDA": LAMBDA, "Vsat": Vsat,"Tbar":Tbar
    }

    DC_modeling.Changeparas(paras)
    datas_input = Getdatas("test_model//5DC_input.txt","Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt","Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt","Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt","Plotname: DC ct1[1]")

    loss = np.average([DC_modeling.inputloss(mds_input,datas_input),DC_modeling.translinloss(mds_trans_lin,datas_trans_lin),
                       DC_modeling.transsubloss(mds_trans_sub,datas_trans_sub),DC_modeling.transloss(mds_trans,datas_trans)])
    return -loss


# 定义参数搜索空间
pbounds = {
    "voff": (-10, 0),
    "nfactor": (0, 5),
    "u0": (1e-3, 0.5),
    "ua": (1e-9, 5e-7),
    "Igsdio": (0.02, 0.5),
    "Njgs": (0.5, 13.5),
    "Igddio": (2.5, 63),
    "Njgd": (0.6, 15.6),
    "Rshg": (2e-4, 5e-3),
    "Eta0": (0.02, 0.52),
    "Vdscale": (1.1, 28.3),
    "Cdscd": (-5, 5),
    "Rsc": (9.2e-5, 2.3e-3),
    "Rdc": (2.4e-4, 6.1e-3),
    "UTE": (-3, -0.2),
    "RTH0": (4.8, 120),
    "LAMBDA": (4e-4, 0.011),
    "Vsat": (5e4, 1.3e6),
    "Tbar": (2e-9, 5.7e-8)
}

# 创建Bayesian优化对象
optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds)

# 进行优化
start_time = time.time()
optimizer.maximize(init_points=10, n_iter=20)
end_time = time.time()
execution_time = end_time - start_time
print("execution_time: ",execution_time)

import matplotlib.pyplot as plt


target_values = [-res['target'] for res in optimizer.res]
iters = np.arange(1,len(target_values)+1,1)
best_values = []
for i in range(len(target_values)):
    best_values.append(min(target_values[:i+1]))

# with open('boundingbox_yes.json', 'w') as f:
#     # Write the data to the file as json
#     json.dump(best_values, f)

plt.plot(best_values,c="red")
plt.scatter(iters,target_values)
plt.xlabel('Iteration')
plt.ylabel('Target Value')
plt.title('GP')
plt.show()


#with open('gp_data.json', 'w') as f:
#    json.dump(best_values, f)

# 输出最优参数和最优结果
best_params = optimizer.max['params']
best_result = - optimizer.max['target']
print(f"Best Params: {best_params}")
print(f"Best Result: {best_result}")

DC_modeling.plots(best_params)
