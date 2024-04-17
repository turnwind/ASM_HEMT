import matplotlib.pyplot as plt
import numpy as np
import re
import json
import time
import optuna
import optuna.visualization as vis
import DC_modeling
import DataExtraction
from DC_modeling import Getdatas
optuna.logging.set_verbosity(optuna.logging.WARNING)



pbounds = {
    "voff": (-10, 0),
    "nfactor": (0, 5),
    "u0": (1e-3, 0.5),
    "ua": (1e-9, 5e-7),
    "igsdio": (0.02, 0.5),
    "njgs": (0.5, 13.5),
    "igddio": (2.5, 63),
    "njgd": (0.6, 15.6),
    "rshg": (2e-4, 5e-3),
    "eta0": (0.02, 0.52),
    "vdscale": (1.1, 28.3),
    "cdscd": (-5, 5),
    "rsc": (9.2e-5, 2.3e-3),
    "rdc": (2.4e-4, 6.1e-3),
    "ute": (-3, -0.2),
    "rth0": (4.8, 120),
    "lambda": (4e-4, 0.011),
    "vsat": (5e4, 1.3e6),
    "tbar": (2e-9, 5.7e-8)
}

mds_input = DataExtraction.loadmeasures(12)
mds_trans_lin = DataExtraction.loadmeasures(7)
mds_trans_sub = DataExtraction.loadmeasures(10)
mds_trans = DataExtraction.loadmeasures(9)
mds_out = DataExtraction.loadmeasures(8)

inivalue = {}
with open("test_model/inivalue.txt","r") as file:
    content = file.read()


for i in pbounds.keys():
    i = i.lower()
    pattern = " "+i + r" =([\d\w\+\-E\.]+)"
    value = re.findall(pattern,content)
    value = float(value[0])
    inivalue[i] = value
    low = min(0.8*value,1.2*value)
    high = max(0.8 * value, 1.2 * value)
    pbounds[i] = (low,high)

#DC_modeling.plots(inivalue)
#print(inivalue)
# flag = [0,0,0,0] - 1 2 4 8
flag = 1
def objective(trial):
    # Define the parameters for the trial
    params = {
        key: trial.suggest_float(key, value[0], value[1])
        #key: trial.suggest_float(key, 0, 200,step=0.01) if key not in specific_ranges else trial.suggest_float(key, *specific_ranges[key])
        for key, value in pbounds.items()
    }
    DC_modeling.Changeparas(params)
    if flag == 1:
        datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    elif flag == 2:
        datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    elif flag == 4:
        datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    elif flag == 8:
        datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")
    else:
        #datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
        #datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
        datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
        datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")

    if flag == 1:
        # loss1 = np.abs(mds_trans_lin[0, :, 1]+datas_trans_lin[0, :, 13])
        # loss2 = np.abs(mds_trans_lin[1, :, 1]+datas_trans_lin[1, :, 13])
        # loss3 = np.abs(mds_trans_lin[2, :, 1]+datas_trans_lin[2, :, 13])
        # l = (loss1+loss2+loss3)/3
        # print(np.round(l,4))
        loss = DC_modeling.translinloss(mds_trans_lin, datas_trans_lin)
    elif flag == 2:
        loss = DC_modeling.inputloss(mds_input, datas_input)
    elif flag == 4:
        loss = DC_modeling.transsubloss(mds_trans_sub, datas_trans_sub)
    elif flag == 8:
        loss =  DC_modeling.transloss(mds_trans, datas_trans)
    else:
        # loss = np.average(
        # [DC_modeling.inputloss(mds_input, datas_input), DC_modeling.translinloss(mds_trans_lin, datas_trans_lin),
        #  DC_modeling.transsubloss(mds_trans_sub, datas_trans_sub), DC_modeling.transloss(mds_trans, datas_trans)])
        loss = np.average(
        [DC_modeling.transsubloss(mds_trans_sub, datas_trans_sub), DC_modeling.transloss(mds_trans, datas_trans)])
    return loss


values = []
def print_params_callback(study, trial):
    params = trial.params
    values.append(trial.values[0])
    #print(f"Iteration: {trial.number}, loss: {trial.values[0]}")

### 1

fixed = {
    'igsdio': inivalue["igsdio"], 'njgs': inivalue["njgs"], 'igddio': inivalue["igddio"], 'njgd': inivalue["njgd"],'rshg': inivalue["rshg"],
    'eta0': inivalue["eta0"], 'vdscale': inivalue["vdscale"], 'cdscd': inivalue["cdscd"],
    'rsc': inivalue["rsc"], 'rdc': inivalue["rdc"], 'ute': inivalue["ute"], 'rth0': inivalue["rth0"],
    'lambda': inivalue["lambda"], 'vsat': inivalue["vsat"], 'tbar': inivalue["tbar"]
}

study1 = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize')
study1.sampler = optuna.samplers.PartialFixedSampler(fixed,study1.sampler)
study1.optimize(objective, n_trials=50, show_progress_bar=True,callbacks=[print_params_callback])
print("period 1: " + str(study1.best_params))
DC_modeling.plotsingle(study1.best_params,1)
# plt.plot(values)
# plt.show()