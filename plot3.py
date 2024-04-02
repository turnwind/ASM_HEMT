import matplotlib.pyplot as plt
import numpy as np

import DataExtraction

mds = DataExtraction.loadmeasures(2)
resfile  = DataExtraction.get_values(flag=0,file="model//3CV_ini.txt")

labels = []
def getdatas(idxstr):
    with open("model//" + resfile, 'r') as file:
        lines = file.readlines()

    start_index = []
    for i, line in enumerate(lines):
        if idxstr in line:
            start_index.append(i)

    nlabel = lines[start_index[0]+2].split(":")[1]
    npoint = lines[start_index[0]+3].split(":")[1]
    labels.append(lines[start_index[0]+4].split("\t")[2])
    for i in range(int(nlabel)-1):
        labels.append(lines[start_index[0] + 5+i].split("\t")[3])

    datas = []
    for i in range(len(start_index)):
        counts = int(npoint)
        j = 0
        values = []
        while counts:
            if 'Values:' in lines[start_index[i]+j]:
                counts -= 1
                for k in range(int(nlabel)//2):
                    line = lines[start_index[i]+j+k+1]
                    line = line.strip()
                    value = line.split("\t")
                    if k == 0:
                        value.pop(0)
                    for d in range(2):
                        values.append(float(value[d]))
                j = k+1+j
            j += 1
        datas.append(values)
    return np.array(datas)

def getdata2(idxstr):
    with open("model//" + resfile, 'r') as file:
        lines = file.readlines()

    start_index = []
    for i, line in enumerate(lines):
        if idxstr in line:
            start_index.append(i)

    nlabel = lines[start_index[0]+2].split(":")[1]
    npoint = lines[start_index[0]+3].split(":")[1]
    labels.append(lines[start_index[0]+4].split("\t")[2])
    for i in range(int(nlabel)-1):
        labels.append(lines[start_index[0] + 5+i].split("\t")[3])

    datas = []
    for i in range(len(start_index)):
        counts = int(npoint)
        j = 0
        values = []
        while counts:
            if 'Values:' in lines[start_index[i]+j]:
                counts -= 1
                for k in range(int(nlabel)):
                    line = lines[start_index[i]+j+k+1]
                    line = line.strip()
                    value = line
                    if k == 0:
                        value = line.split("\t")[1]
                    value = value.split(",")
                    values.append([float(i) for i in value])
                j = k+1+j
            j += 1
        datas.append(values)
    return np.array(datas)

datas1 = getdatas(idxstr="Plotname: DC dcct2[1].dcct1[1/3]")
datas2 = getdatas(idxstr="Plotname: DC dcct2[1].dcct1[2/3]")
datas3 = getdatas(idxstr="Plotname: DC dcct2[1].dcct1[3/3]")

ds1 = getdata2(idxstr="Plotname: SP ct2[1].ct1[1/3]")
ds2 = getdata2(idxstr="Plotname: SP ct2[1].ct1[2/3]")
ds3 = getdata2(idxstr="Plotname: SP ct2[1].ct1[3/3]")

def s_to_y(Z0, S):
    #print(S)
    Y0 = 1 / Z0
    S_com = ((1 + S[0, 0]) * (1 + S[1, 1])) - (S[0, 1] * S[1, 0])
    #print(S_com)
    Y = np.zeros((2, 2), dtype=np.complex128)
    Y[0, 0] = Y0 * ((1 - S[0, 0]) * (1 + S[1, 1]) + S[0, 1] * S[1, 0]) / S_com
    Y[0, 1] = Y0*-2 * S[0, 1] / S_com
    Y[1, 0] = Y0*-2 * S[1, 0] / S_com
    Y[1, 1] = Y0 * ((1 + S[0, 0]) * (1 - S[1, 1]) + S[0, 1] * S[1, 0]) / S_com

    return Y

def convert(ds):
    S11 = ds[:,1]
    S12 = ds[:,2]
    S21 = ds[:,3]
    S22 = ds[:,4]

    S11s = S11[:, 0] + S11[:, 1] * 1j
    S12s = S12[:, 0] + S12[:, 1] * 1j
    S21s = S21[:, 0] + S21[:, 1] * 1j
    S22s = S22[:, 0] + S22[:, 1] * 1j


    s_matrixs = []
    for i in range(len(S11s)):
        s_matrixs.append(np.array([[S11s[i],S12s[i]],[S21s[i],S22s[i]]]))
    #print(s_matrixs)


    Y_matrixs = np.empty_like(s_matrixs)
    for i in range(len(s_matrixs)):
        Y_matrixs[i]= s_to_y(50,s_matrixs[i])

    return Y_matrixs

Y1 = convert(ds1)
Y2 = convert(ds2)
Y3 = convert(ds3)
# Y11 = Y_matrixs[:,0,0]
# Y12 = Y_matrixs[:,0,1]
# Y21 = Y_matrixs[:,1,0]
# Y22 = Y_matrixs[:,1,1]

def toC(Y):
    Cgs = (np.imag(Y[:,0,0]) + np.imag(Y[:,0,1]))/(2*np.pi*0.1) *1e3
    Cgd = (-np.imag(Y[:,0,1])) / (2 * np.pi * 0.1)
    Cds = (np.imag(Y[:,1,1]) + np.imag(Y[:,0,1])) / (2 * np.pi * 0.1)
    return  Cgs,Cgd,Cds

xx = np.arange(-9,0.2,0.2)
#for i in range(int(len(labels)//1)):
for i in range(1):
    cgs1,cgd1,cds1 = toC(Y1)
    cgs2, cgd2, cds2 = toC(Y2)
    cgs3, cgd3, cds3 = toC(Y3)
    # plt.plot(xx,cgs1)
    # plt.plot(xx, cgs2)
    # plt.plot(xx, cgs3)

    figure, axes = plt.subplots(2, 2)

    axes[0][0].plot(xx,cgs1,label = "0mv")
    axes[0][0].plot(xx,cgs2, label = "200mv")
    axes[0][0].plot(xx,cgs3, label = "400mv")

    axes[0][1].plot(xx,cgd1,label = "0mv")
    axes[0][1].plot(xx,cgd2, label = "200mv")
    axes[0][1].plot(xx,cgd3, label = "400mv")

    axes[1][0].plot(xx,cds1,label = "0mv")
    axes[1][0].plot(xx,cds2, label = "200mv")
    axes[1][0].plot(xx,cds3, label = "400mv")
    #axes[0][0].plot(mds[0,:,0],mds[0,:,1],label = "m_50mv",marker = "o",c="red")
    #axes[0][0].plot(mds[1,:,0],mds[1,:,1], label = "m_100mv",marker = "o",c="red")
    #axes[0][0].plot(mds[2,:,0],mds[2,:,1], label = "m_150mv",marker = "o",c="red")

    plt.xlabel("Vdc")
    plt.ylabel(labels[i])
    plt.legend()
    plt.show()

