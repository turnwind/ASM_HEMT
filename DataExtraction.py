import os
import subprocess
import json
import numpy as np

def run_script(filename):
    resfile = "res_" + filename
    script = r'C:\Users\87515\Documents\PyProject\ASM_HEMT\test_model\script.bat'

    with open(script, 'r') as file:
        data = file.readlines()

    data[-1] = 'hpeesofsim --MTS_enabled  -r [] {}\n'
    data[-1] = data[-1].replace("[]",resfile).replace("{}", filename)

    with open(script, 'w') as file:
        file.writelines(data)

    script_dir = os.path.dirname(script)
    subprocess.run(["cmd", "/c", script], cwd=script_dir,stdout=subprocess.DEVNULL)

def get_values(line_len = 8,numbers = 97, nv = 14,flag = 1,file=None):

    filename =  file.split("//")[-1]
    resfile = "res_" + filename
    run_script(filename)


    if flag != 1:
        return resfile
    datas = []
    with open("test_model//"+resfile, 'r') as file:
        lines = file.readlines()

    # 查找起始行索引
    start_index = []
    labelidx = 0
    for i, line in enumerate(lines):
        if line.strip().split('\t')[0] == "Variables:" and not labelidx:
            labelidx  = i
        if line.strip() == 'Values:':
            start_index.append(i+1)

    labels = []
    for i in range(nv):
        line = lines[labelidx+i].strip().split('\t')
        if i == 0:
            labels.append(line[2])
        else:
            labels.append(line[1])


    # 提取数据行
    for k in range(len(start_index)-1):
        data_lines = lines[start_index[k]:]
        datas.append([])
        for i in range(numbers):
            values = []
            for j in range(7):
                line = data_lines[i * line_len + j]
                line = line.strip()
                value = line.split("\t")
                if j == 0:
                    value.pop(0)
                for d in range(2):
                    values.append(float(value[d]))
            datas[k].append(values)
    return np.array(datas),labels

def get_file_paths(directory):
    file_paths = []  # 创建一个空列表来储存文件路径
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            full_path = os.path.join(dirpath, file)  # 使用os.path.join()来合并获取完整的文件路径
            file_paths.append(full_path.replace("\\", "/"))
    return file_paths


def loadmeasures(idx):
    #file = r"C:\Users\87515\model\Avail_Meas_Data\Data_for_Modeling\_DeviceA_5053_8x100~DC_MODELING_vt_u0~id_vgs__Transfer__lin.mdm"
    directory = 'Avail_Meas_Data/Data_for_Modeling'
    files = get_file_paths(directory)
    # for i in range(len(files)):
    #     print("{} : {}".format(i,files[i]))
    file = files[idx]
    datas = []
    with open(file, 'r') as file:
        lines = file.readlines()

    start_index = []
    for i, line in enumerate(lines):
        line = line.strip().split('\t')[0]
        if line and line[0] == '#':
            start_index.append(i + 1)

    for i in range(len(start_index)):
        j = 0
        ds = lines[start_index[i]:]
        datas.append([])
        while ds[j].strip() != "END_DB":
            line = ds[j].strip().split(" ")
            values = []
            for d in range(len(line)):
                if line[d] != "":
                    values.append(float(line[d]))
            datas[i].append(values)
            j += 1

    return  np.array(datas)