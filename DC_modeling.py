import numpy as np
import DataExtraction
import re
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#---------> DC_modeling
def Extracdatas(idxstr,resfile):
    with open("test_model//" + resfile, 'r') as file:
        lines = file.readlines()

    start_index = []
    for i, line in enumerate(lines):
        if idxstr in line:
            start_index.append(i)
    labels = []
    nlabel = lines[start_index[0]+2].split(":")[1]
    nlabel = int(nlabel)
    npoint = lines[start_index[0]+3].split(":")[1]
    labels.append(lines[start_index[0]+4].split("\t")[2])
    for i in range(int(nlabel)-1):
        labels.append(lines[start_index[0] + 5+i].split("\t")[3])

    datas = []
    for k in range(len(start_index)):
        data_lines = lines[start_index[k]+5+nlabel:]
        datas.append([])
        for i in range(int(npoint)):
            values = []
            for j in range(nlabel//2):
                line = data_lines[i * (nlabel//2+1) + j]
                line = line.strip()
                value = line.split("\t")
                if j == 0:
                    value.pop(0)
                for d in range(2):
                    values.append(float(value[d]))
            datas[k].append(values),labels
    return np.array(datas)

def Getdatas(file,idxstr):
    res_file = DataExtraction.get_values(file=file, flag=0)
    datas = Extracdatas(idxstr,res_file)
    return datas

######paras
DC_paras = ["voff", "nfactor", "u0", "ua", "Igsdio", "Njgs", "Igddio", "Njgd", "Rshg", "Eta0", "Vdscale", "Cdscd",
            "Rsc", "Rdc", "UTE", "RTH0", "LAMBDA", "Vsat", "Tbar"]

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

dc_netfiles = ["5DC_input.txt","5DC_transfer_lin.txt","5DC_trans_su.txt","5DC_trans.txt","5DC_output.txt"]

mds_input = DataExtraction.loadmeasures(12)
mds_trans_lin = DataExtraction.loadmeasures(7)
mds_trans_sub = DataExtraction.loadmeasures(10)
mds_trans = DataExtraction.loadmeasures(9)
mds_out = DataExtraction.loadmeasures(8)

def generate_paras(pbounds):
    paras = {}
    for key, (low, high) in pbounds.items():
        paras[key] = np.random.uniform(low, high)
    return paras

def Changeparas(paras):
    #dc_netfiles = ["testnet.txt"]
    for i in range(len(dc_netfiles)-1):
        with open("test_model//"+dc_netfiles[i], 'r+') as file:
            content = file.read()
        # 替换参数值
        for parameter, new_value in paras.items():
            parameter = parameter.lower()
            new_value = float('%.4g' % new_value)
            new_value = str(new_value).upper()
            content = re.sub(f'{parameter} =[\d\w\+\-E\.]*', f'{parameter} ={new_value}', content, flags=re.IGNORECASE)

        # 将修改后的内容写回文件
        with open("test_model//"+dc_netfiles[i], 'w') as file:
            file.write(content)

        #DataExtraction.run_script(dc_netfiles[i])


######loss
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))


def inputloss(mds,ds):

    loss1 = RMSE(mds[0,-10:,1],-ds[0,-10:,13])
    loss2 = RMSE(mds[0, -10:, 2], -ds[0,-10 :, 12])*4
    return  np.average([loss1,loss2])

def translinloss(mds,ds):
     loss1 = RMSE(mds[0,:,1],-ds[0,:,13])
     loss2 = RMSE(mds[1, :, 1], -ds[1, :, 13])
     loss3 = RMSE(mds[2, :, 1], -ds[2, :, 13])
     return np.average([loss1, loss2,loss3])

def transsubloss(mds, ds):
    losses = []
    for j in range(21):
        losses.append( RMSE(mds[j, :, 1], -ds[j, :, 13]) )
    return np.average(losses)


def transloss(mds,ds):
    losses = []
    for j in range(6):
        losses.append( RMSE(mds[j, :, 1], -ds[j, :, 13]) )
    for k in range(len(ds[:])):
        losses.append(RMSE(mds[:, k, 1],-ds[:,k,13]))
    return np.average(losses)

def outloss(mds,ds):
    losses = []
    for j in range(24):
        losses.append( RMSE(mds[j, :, 1], -ds[j, :, 13]) )

    return np.average(losses)

def plotsingle(paras,flag):
    Changeparas(paras)

    if flag == 1:
        datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    elif flag == 2:
        datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    elif flag == 4:
        datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    elif flag == 8:
        datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")
    else:
        datas_out = Getdatas("test_model//5DC_output.txt", "Plotname: DC ct1[1]")

    if flag == 1:
        figure = plt.figure(figsize=(12, 6))
        plt.plot(datas_trans_lin[0, :, 0], -datas_trans_lin[0, :, 13], label="50mv", c="blue")
        plt.plot(datas_trans_lin[1, :, 0], -datas_trans_lin[1, :, 13], label="100mv", c="blue")
        plt.plot(datas_trans_lin[2, :, 0], -datas_trans_lin[2, :, 13], label="150mv", c="blue")
        plt.plot(mds_trans_lin[0, :, 0], mds_trans_lin[0, :, 1], label="m_50mv", marker="o", c="red")
        plt.plot(mds_trans_lin[1, :, 0], mds_trans_lin[1, :, 1], label="m_100mv", marker="o", c="red")
        plt.plot(mds_trans_lin[2, :, 0], mds_trans_lin[2, :, 1], label="m_150mv", marker="o", c="red")
        plt.title("Transfer_lin")
        plt.xlabel("Vg")
        plt.ylabel("Id")
    elif flag ==2:
        figure, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].plot(datas_input[0, :, 0], -datas_input[0, :, 13], c="blue")
        axes[1].plot(datas_input[0, :, 0], -datas_input[0, :, 12], c="blue")
        axes[0].plot(mds_input[0, :, 0], mds_input[0, :, 1], marker="o", c="red")
        axes[1].plot(mds_input[0, :, 0], mds_input[0, :, 2], marker="o", c="red")
        axes[0].set_xlabel("vg")
        axes[0].set_ylabel("ig")
        axes[1].set_xlabel("vg")
        axes[1].set_ylabel("id")
        plt.suptitle('ig_vgs__Input')
    elif flag == 4:
        figure = plt.figure(figsize=(12, 6))
        for j in range(21):
            plt.plot(datas_trans_sub[0, :, 0], -datas_trans_sub[j, :, 13], c="blue")
            plt.plot(mds_trans_sub[0, :, 0], mds_trans_sub[j, :, 1], marker="o", c="red", markersize=2)
            plt.xlabel("Vg")
            plt.ylabel("Ig")
            plt.title("id_vgs__Transfer_subVOFF")
    elif flag == 8:
        figure, axes = plt.subplots(1, 2, figsize=(12, 6))
        for j in range(6):
            axes[0].plot(datas_trans[0, :, 0], -datas_trans[j, :, 13], c="blue")
            axes[0].plot(mds_trans[0, :, 0], mds_trans[j, :, 1], marker="o", c="red", markersize=2)
        xx = np.arange(0.1, 24.1, 4)
        for k in range(len(datas_trans[:])):
            axes[1].plot(xx, -datas_trans[:, k, 13], c="blue")
            axes[1].plot(xx, mds_trans[:, k, 1], marker="o", c="red", markersize=2)
        axes[0].set_xlabel("vg")
        axes[0].set_ylabel("id")
        axes[1].set_xlabel("vd")
        axes[1].set_ylabel("id")
        plt.suptitle('ig_vgs__Transfer')
    plt.show()

def plots(paras):
    figure, axes = plt.subplots(2, 3, figsize=(12, 6))
    Changeparas(paras)
    # 更新数据
    datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")
    # datas_out = Getdatas("test_model//5DC_output.txt", "Plotname: DC ct1[1]")

    # 绘制新的帧
    #### input
    axes[0][0].plot(datas_input[0, :, 0], -datas_input[0, :, 13], c="blue")
    axes[0][1].plot(datas_input[0, :, 0], -datas_input[0, :, 12], c="blue")
    axes[0][0].plot(mds_input[0, :, 0], mds_input[0, :, 1], marker="o", c="red")
    axes[0][1].plot(mds_input[0, :, 0], mds_input[0, :, 2], marker="o", c="red")

    # ### lin
    axes[1][0].plot(datas_trans_lin[0, :, 0], -datas_trans_lin[0, :, 13], label="50mv")
    axes[1][0].plot(datas_trans_lin[1, :, 0], -datas_trans_lin[1, :, 13], label="100mv")
    axes[1][0].plot(datas_trans_lin[2, :, 0], -datas_trans_lin[2, :, 13], label="150mv")
    axes[1][0].plot(mds_trans_lin[0, :, 0], mds_trans_lin[0, :, 1], label="m_50mv", marker="o", c="red")
    axes[1][0].plot(mds_trans_lin[1, :, 0], mds_trans_lin[1, :, 1], label="m_100mv", marker="o", c="red")
    axes[1][0].plot(mds_trans_lin[2, :, 0], mds_trans_lin[2, :, 1], label="m_150mv", marker="o", c="red")
    #
    # ### sub
    for j in range(21):
        axes[1][1].plot(datas_trans_sub[0, :, 0], -datas_trans_sub[j, :, 13], c="blue")
        axes[1][1].plot(mds_trans_sub[0, :, 0], mds_trans_sub[j, :, 1], marker="o", c="red", markersize=2)

    # ### trans
    for j in range(6):
        axes[0][2].plot(datas_trans[0, :, 0], -datas_trans[j, :, 13], c="blue")
        axes[0][2].plot(mds_trans[0, :, 0], mds_trans[j, :, 1], marker="o", c="red", markersize=2)
    xx = np.arange(0.1, 24.1, 4)
    for k in range(len(datas_trans[:])):
        axes[1][2].plot(xx, -datas_trans[:, k, 13], c="blue")
        axes[1][2].plot(xx, mds_trans[:, k, 1], marker="o", c="red", markersize=2)
    #
    # for j in range(24):
    #     axes[1][2].plot(datas_out[0,:,0],-datas_out[j,:,13],c ="blue")
    #     axes[1][2].plot(mds_out[0, :, 0], mds_out[j, :, 1], marker="o", c="red", markersize=2)
    plt.show()




def update(i):
    # 更新数据
    datas_input = Getdatas("test_model//5DC_input.txt", "Plotname: DC dc1[1]")
    datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt", "Plotname: DC ct1[1]")
    datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt", "Plotname: DC ct1[1]")
    datas_trans = Getdatas("test_model//5DC_trans.txt", "Plotname: DC ct1[1]")
    datas_out = Getdatas("test_model//5DC_output.txt", "Plotname: DC ct1[1]")

    # 清除上一帧
    for k in range(2):
        for j in range(4):
            axes[k][j].clear()
    # 绘制新的帧
    #### input
    axes[0][0].plot(datas_input[0, :, 0], -datas_input[0, :, 13], c="blue")
    axes[0][1].plot(datas_input[0, :, 0], -datas_input[0, :, 12], c="blue")
    axes[0][0].plot(mds_input[0, :, 0], mds_input[0, :, 1], marker="o", c="red")
    axes[0][1].plot(mds_input[0, :, 0], mds_input[0, :, 2], marker="o", c="red")

    # ### lin
    axes[1][0].plot(datas_trans_lin[0,:,0],-datas_trans_lin[0,:,13],label = "50mv")
    axes[1][0].plot(datas_trans_lin[1,:,0],-datas_trans_lin[1,:,13], label = "100mv")
    axes[1][0].plot(datas_trans_lin[2,:,0],-datas_trans_lin[2,:,13], label = "150mv")
    axes[1][0].plot(mds_trans_lin[0,:,0],mds_trans_lin[0,:,1],label = "m_50mv",marker = "o",c="red")
    axes[1][0].plot(mds_trans_lin[1,:,0],mds_trans_lin[1,:,1], label = "m_100mv",marker = "o",c="red")
    axes[1][0].plot(mds_trans_lin[2,:,0],mds_trans_lin[2,:,1], label = "m_150mv",marker = "o",c="red")
    #
    # ### sub
    for j in range(21):
        axes[1][1].plot(datas_trans_sub[0,:,0],-datas_trans_sub[j,:,13],c ="blue")
        axes[1][1].plot(mds_trans_sub[0, :, 0], mds_trans_sub[j, :, 1], marker="o", c="red", markersize=2)

    # ### trans
    for j in range(6):
        axes[0][2].plot(datas_trans[0,:,0],-datas_trans[j,:,13],c ="blue")
        axes[0][2].plot(mds_trans[0, :, 0], mds_trans[j, :, 1], marker="o", c="red", markersize=2)
    xx = np.arange(0.1, 24.1, 4)
    for k in range(len(datas_trans[:])):
        axes[1][2].plot(xx,-datas_trans[:,k,13],c ="blue")
        axes[1][2].plot(xx, mds_trans[:, k, 1], marker="o", c="red", markersize=2)

    for j in range(24):
         axes[0][3].plot(datas_out[0,:,0],-datas_out[j,:,13],c ="blue")
         axes[0][3].plot(mds_out[0, :, 0], mds_out[j, :, 1], marker="o", c="red", markersize=2)

    paras = generate_paras(pbounds)
    Changeparas(paras)
    print(i)

    #return  datas_input[0, :, 0], -datas_input[0, :, 13]

if __name__ == "__main__":


    #
    #
    # i = 0
    # while True:
    #     datas_input = Getdatas("test_model//5DC_input.txt","Plotname: DC dc1[1]")
    #     datas_trans_lin = Getdatas("test_model//5DC_transfer_lin.txt","Plotname: DC ct1[1]")
    #     datas_trans_sub = Getdatas("test_model//5DC_trans_su.txt","Plotname: DC ct1[1]")
    #     datas_trans = Getdatas("test_model//5DC_trans.txt","Plotname: DC ct1[1]")
    #     #datas_out = Getdatas("test_model//5DC_output.txt","Plotname: DC ct1[1]")
    #     print(i)
    #     i = i+1
    #     paras = generate_paras(pbounds)
    #     Changeparas(paras)
    #
    #     print(inputloss(mds_input,datas_input))
    #     print(translinloss(mds_trans_lin,datas_trans_lin))
    #     print(transsubloss(mds_trans_sub,datas_trans_sub))
    #     print(transloss(mds_trans,datas_trans))
    #     #print(outloss(mds_out,datas_out))


    figure, axes = plt.subplots(2, 4, figsize = (12,6))
    ani = animation.FuncAnimation(figure, update, frames=range(100), repeat=True)
    plt.show()