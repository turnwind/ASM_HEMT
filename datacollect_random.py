import  DC_modeling
from DC_modeling import Getdatas

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

iters = 1
for i in range(iters):
    paras = DC_modeling.generate_paras(pbounds)
    DC_modeling.Changeparas(paras)
    datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")
    datas_out = Getdatas("test_model//5DC_output.txt", "Plotname: DC ct1[1]")
    datas_input = -datas_input[0,:,12:]
    datas_trans_lin = -datas_trans_lin[:3,:,13]
    dssub = []
    for j in range(21):
        dssub.append(-datas_trans_sub[j, :, 13])
    dstrans = []
    for k in range(6):
        dstrans.append([-datas_trans[k, :, 13]])
    for k in range(len(datas_trans[:])):
        dstrans.append([-datas_trans[:,k,13]])
    dsout = []
    for j in range(24):
        dsout.append(-datas_out[j, :, 13])

    datasets = [datas_input, datas_trans_lin, dssub, dstrans, dsout]
    filenames = ['ds_input.txt', 'ds_trans_lin.txt', 'ds_sub.txt', 'ds_trans.txt', 'ds_out.txt']
    for data, filename in zip(datasets, filenames):
        filename = "Datafortranin//" +filename
        # 'w'表示write，会覆盖原有内容；如果你希望追加内容，可以使用'a'
        with open(filename, 'w') as file:
            for item in data:
                # write()函数只接受字符串类型的参数，所以我们需要将数据转化为字符串
                file.write("%s\n" % item)