import  DC_modeling
import  json
from DC_modeling import Getdatas


def generate_paras(pbounds):
    paras = {}
    for key, (low, high) in pbounds.items():
        paras[key] = np.random.uniform(low, high)
    return paras

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

#设置随机数种子
import numpy as np
import time
np.random.seed(int(time.time()))

from tqdm import tqdm

iters = 10000

for i in tqdm(range(iters+1, iters+10001)):
    paras = generate_paras(pbounds)
    DC_modeling.Changeparas(paras)
    datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")
    datas_out = Getdatas("test_model//5DC_output.txt", "Plotname: DC ct1[1]")
    datas_input = -datas_input[0,:,12:]
    datas_input = datas_input.tolist()
    datas_trans_lin = -datas_trans_lin[:3,:,13]
    datas_trans_lin = datas_trans_lin.tolist()
    dssub = []
    for j in range(21):
        dssub.append((-datas_trans_sub[j, :, 13]).tolist())
    dstrans = []
    for k in range(6):
        dstrans.append((-datas_trans[k, :, 13]).tolist())
    for k in range(len(datas_trans[:])):
        dstrans.append((-datas_trans[:,k,13]).tolist())
    dsout = []
    for j in range(24):
        dsout.append((-datas_out[j, :, 13]).tolist())

    datasets = [datas_input, datas_trans_lin, dssub, dstrans, dsout]
    # filenames = ['ds_input{}.txt', 'ds_trans_lin{}.txt', 'ds_sub{}.txt', 'ds_trans{}.txt', 'ds_out{}.txt']
    filenames = ['ds_input{}.json', 'ds_trans_lin{}.json', 'ds_sub{}.json', 'ds_trans{}.json', 'ds_out{}.json']

    for data, filename in zip(datasets, filenames):
        filename = filename.format(i)
        filename = "Datafortranin//" + filename
        # 'w'表示write，会覆盖原有内容；如果你希望追加内容，可以使用'a'
        # with open(filename, 'w') as file:
        #     for item in data:
        #         # write()函数只接受字符串类型的参数，所以我们需要将数据转化为字符串
        #         file.write("%s\n" % item)
        with open(filename, 'w') as f:
            json.dump(data, f)
    try:
        with open('Datafortranin//paras.json', 'r') as f:

            data = json.load(f)
        data.append(paras)
        with open('Datafortranin//paras.json', 'w') as f:
            json.dump(data, f)
    except:
        with open('Datafortranin//paras.json', 'w') as f:
            json.dump([paras], f)