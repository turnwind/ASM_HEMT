import matplotlib.pyplot as plt
import numpy as np

import DataExtraction

mds = DataExtraction.loadmeasures(9)
resfile = DataExtraction.get_values(file="model//5DC_trans.txt",flag=0)

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
    figure, axes = plt.subplots(1, 2)
    for j in range(6):
        axes[0].plot(datas[0,:,0],-datas[j,:,13],c ="blue")
        axes[0].plot(mds[0, :, 0], mds[j, :, 1], marker="o", c="red", markersize=2)

    for k in range(len(datas[:])):
        axes[1].plot(xx,-datas[:,k,13],c ="blue")
        axes[1].plot(xx, mds[:, k, 1], marker="o", c="red", markersize=2)

    plt.xlabel(labels[0])
    plt.ylabel(labels[13])
    plt.legend()
    plt.show()

