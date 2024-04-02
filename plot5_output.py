import matplotlib.pyplot as plt
import numpy as np

import DataExtraction

mds = DataExtraction.loadmeasures(8)
resfile = DataExtraction.get_values(file="model//5DC_output.txt",flag=0)

labels = []

def getdatas(idxstr):
    with open("model//" + resfile, 'r') as file:
        lines = file.readlines()

    start_index = []
    for i, line in enumerate(lines):
        if idxstr in line:
            start_index.append(i)

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
            datas[k].append(values)

    return np.array(datas)

datas = getdatas("Plotname: DC ct1[1]")

xx = np.arange(0.1,24.1,4)

for i in range(1):
    figure, axes = plt.subplots(1, 1)
    for j in range(24):
        axes.plot(datas[0,:,0],-datas[j,:,13],c ="blue")
        axes.plot(mds[0, :, 0], mds[j, :, 1], marker="o", c="red", markersize=2)

    # for k in range(len(datas[:])):
    #     axes[1].plot(xx,-datas[:,k,13],c ="blue")
    #     axes[1].plot(xx, mds[:, k, 1], marker="o", c="red", markersize=2)
    # axes[0][0].plot(datas[2,:,0],-datas[2,:,i], label = "150mv")
    # axes[0][0].plot(mds[0,:,0],mds[0,:,1],label = "m_50mv",marker = "o",c="red")
    # axes[0][1].plot(mds[0,:,0],mds[0,:,2], label = "m_100mv",marker = "o",c="red")
    # axes[0][0].plot(mds[2,:,0],mds[2,:,1], label = "m_150mv",marker = "o",c="red")
    plt.xlabel(labels[0])
    plt.ylabel(labels[13])
    plt.legend()
    plt.show()

