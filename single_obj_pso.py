from pyswarm import pso
import time
import DC_modeling
import DataExtraction
import  matplotlib.pyplot as plt
from DC_modeling import Getdatas
import numpy as np

mds_input = DataExtraction.loadmeasures(12)
mds_trans_lin = DataExtraction.loadmeasures(7)
mds_trans_sub = DataExtraction.loadmeasures(10)
mds_trans = DataExtraction.loadmeasures(9)
mds_out = DataExtraction.loadmeasures(8)

mds_input[0, -10:, 1] *= 20
mds_input[0, -10:, 2] *= 20

def objective_function(x):
    paras = {
        "voff": x[0], "nfactor": x[1], "u0": x[2], "ua": x[3], "Igsdio": x[4], "Njgs": x[5],
        "Igddio": x[6], "Njgd": x[7], "Rshg": x[8], "Eta0": x[9], "Vdscale": x[10],
        "Cdscd": x[11], "Rsc": x[12], "Rdc": x[13], "UTE": x[14], "RTH0": x[15], "LAMBDA": x[16], "Vsat": x[17],"Tbar":x[18]
    }
    DC_modeling.Changeparas(paras)
    datas_input = Getdatas("test_model//5DC_input.txt","Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt","Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt","Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt","Plotname: DC ct1[1]")

    loss = np.average([DC_modeling.inputloss(mds_input,datas_input),DC_modeling.translinloss(mds_trans_lin,datas_trans_lin),
                       DC_modeling.transsubloss(mds_trans_sub,datas_trans_sub),DC_modeling.transloss(mds_trans,datas_trans)])
    return loss

# Define the lower and upper bounds for each parameter
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

objective_values = []
lb = [value[0] for value in pbounds.values()]
ub = [value[1] for value in pbounds.values()]
# Use PSO to minimize the objective function
start_time = time.time()
x, fopt,losses = pso(objective_function, lb, ub,maxiter=20,swarmsize=20,debug=True)
end_time = time.time()
execution_time = end_time - start_time
print("execution_time: ",execution_time)


best_values = []
for i in range(len(losses)):
    best_values.append(min(losses[:i+1]))


iters = np.arange(1,len(losses)+1,1)
plt.plot(iters,best_values,c="red")
plt.scatter(iters,losses)
plt.xlabel('Iteration')
plt.ylabel('Target Value')
plt.title('pso')
plt.show()

# Print the optimized parameters
bestparas = {
    "voff": x[0], "nfactor": x[1], "u0": x[2], "ua": x[3], "Igsdio": x[4], "Njgs": x[5],
    "Igddio": x[6], "Njgd": x[7], "Rshg": x[8], "Eta0": x[9], "Vdscale": x[10],
    "Cdscd": x[11], "Rsc": x[12], "Rdc": x[13], "UTE": x[14], "RTH0": x[15], "LAMBDA": x[16], "Vsat": x[17],
    "Tbar": x[18]
}

DC_modeling.plots(bestparas)
