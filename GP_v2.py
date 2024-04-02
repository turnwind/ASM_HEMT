import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Bayesian_optimization.acq import UCB, EI, find_next_batch, optimize_acqf
import matplotlib.pyplot as plt
import sys
import Bayesian_optimization.kernel as kernel
from Bayesian_optimization.cigp import CIGP_withMean
import DC_modeling
import DataExtraction
from DC_modeling import Getdatas

bounds = {
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

def objective(x):
    # Define the parameters for the trial
    paras = {
        "voff": x[0], "nfactor": x[1], "u0": x[2], "ua": x[3], "Igsdio": x[4], "Njgs": x[5],
        "Igddio": x[6], "Njgd": x[7], "Rshg": x[8], "Eta0": x[9], "Vdscale": x[10],
        "Cdscd": x[11], "Rsc": x[12], "Rdc": x[13], "UTE": x[14], "RTH0": x[15], "LAMBDA": x[16], "Vsat": x[17],"Tbar":x[18]
    }
    DC_modeling.Changeparas(paras)
    datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")

    loss = np.average(
        [DC_modeling.inputloss(mds_input, datas_input), DC_modeling.translinloss(mds_trans_lin, datas_trans_lin),
         DC_modeling.transsubloss(mds_trans_sub, datas_trans_sub), DC_modeling.transloss(mds_trans, datas_trans)])
    return loss

train_x = []
train_y = []
n_initial_points = 3
for _ in range(n_initial_points):
    res_para = []
    for i in bounds.keys():
        res_para.append(np.random.uniform(bounds[i][0], bounds[i][1]))
    train_y.append(objective(res_para))
    train_x.append(res_para)

train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y).reshape(-1,1).to(torch.float32)
#kernel1 = kernel.ARDKernel(1,3.0,1.0)
# kernel1 = kernel.MaternKernel(1)
# kernel1 = kernel.LinearKernel(1,-1.0,1.)
kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
model = CIGP_withMean(1, 1, kernel=kernel1, noise_variance=4.)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

def gp_pred(X):
    model.eval()
    with torch.no_grad():
        mean, var = model.forward(train_x, train_y, X)
        return mean,var

ucb = UCB(gp_pred, kappa=5)
ei = EI(gp_pred)
best_y = []

pbounds = []
for i in bounds.keys():
    pbounds.append(bounds[i])
pbounds = np.array(pbounds)

for iteration in range(10):  # Run for 5 iterations

    for i in range(100):
        optimizer.zero_grad()
        loss = -model.log_likelihood(train_x, train_y)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))


    #batch_points = find_next_batch(ucb, bounds, batch_size=1, n_samples=500, f_best=train_x[np.argmax(train_y)])
    # batch_points = ei.find_next_batch(bounds, batch_size=1, n_samples=1000, f_best=train_x[np.argmax(train_y)])
    #find_next_batch(acq)
    batch_points = optimize_acqf(acq=ucb, raw_samples=50, bounds=pbounds, f_best=0, num_restarts=30, options=None)
    batch_points = torch.tensor(batch_points).float()

    # Evaluate the objective function
    new_y = objective(batch_points.squeeze()).reshape(-1,1)

    # Update the model
    train_x = torch.cat([train_x, batch_points])
    train_y = torch.cat([train_y, new_y])
    # Store the best objective value found so far
    best_y.append(new_y.max().item())
    # Visualization

    # 在关键迭代时保存模型预测
    # if (iteration + 1) in key_iterations:
    #     model.eval()
    #     fixed_dims = torch.full((1, input_dim - 1), 5.0)  # Example: set them to the midpoint (5.0)
    #     test_points = torch.linspace(0, 10, 100)
    #     test_X = torch.cat((test_points.unsqueeze(1), fixed_dims.expand(test_points.size(0), -1)), 1)
    #     true_y = objective_function(test_X)
    #
    #     with torch.no_grad():
    #         pred_mean, pred_std = model.forward(train_x, train_y, test_X)
    #         predictions.append((pred_mean, pred_std))