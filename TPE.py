import os
import re
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import json
import time
import optuna
import optuna.visualization as vis
import logging
import DC_modeling
import DataExtraction
from DC_modeling import Getdatas


optuna.logging.set_verbosity(optuna.logging.WARNING)



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

mds_input = DataExtraction.loadmeasures(12)
mds_trans_lin = DataExtraction.loadmeasures(7)
mds_trans_sub = DataExtraction.loadmeasures(10)
mds_trans = DataExtraction.loadmeasures(9)
mds_out = DataExtraction.loadmeasures(8)


def objective(trial):
    # Define the parameters for the trial
    params = {
        key: trial.suggest_float(key, value[0], value[1])
        #key: trial.suggest_float(key, 0, 200,step=0.01) if key not in specific_ranges else trial.suggest_float(key, *specific_ranges[key])
        for key, value in pbounds.items()
    }
    DC_modeling.Changeparas(params)
    datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")

    loss = np.average(
        [DC_modeling.inputloss(mds_input, datas_input), DC_modeling.translinloss(mds_trans_lin, datas_trans_lin),
         DC_modeling.transsubloss(mds_trans_sub, datas_trans_sub), DC_modeling.transloss(mds_trans, datas_trans)])
    return loss


values = []
def print_params_callback(study, trial):
    # 获取当前试验的参数值
    params = trial.params
    values.append(trial.values[0])
    #print(f"Iteration: {trial.number}, loss: {trial.values[0]}")

study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize')
study.optimize(objective, n_trials=400, show_progress_bar=True,callbacks=[print_params_callback])

trials_data = study.trials
# Print the best parameters
print(study.best_params)

with open('tpeiters.json', 'w') as f:
     json.dump(values, f)

plt.plot(values)
plt.show()


vis.plot_optimization_history(study).show()

# Plot the importance of parameters
vis.plot_param_importances(study).show()

#vis.plot_contour(study, params=['Rg', 'Rd', 'Rs']).show()
# Plot the slice of parameters
#vis.plot_slice(study, params=['Rg', 'Rd', 'Rs']).show()

DC_modeling.plots(study.best_params)
