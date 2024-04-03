import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

filenames = ['ds_input{}.json', 'ds_trans_lin{}.json', 'ds_sub{}.json', 'ds_trans{}.json', 'ds_out{}.json']

##### data
dataY = []
dataX = []

with open("Datafortranin//paras2.json") as f:
    dataY = json.load(f)
dataY = np.array(dataY)
dataY = dataY[1:,:]

# end = 19533
# for i in range(10002,end):
#     filename = filenames[-1].format(i)
#     filename = "Datafortranin//" + filename
#     data = []
#     with open(filename,"r") as f:
#         data = np.array(json.load(f))
#     dataX.append(data)
# dataX = np.array(dataX)
with open("Datafortranin//X2.json") as f:
    dataX = json.load(f)
dataX = np.array(dataX)

# dataX = np.transpose(dataX, (2, 0, 1))
#dataY = dataY.reshape(-1, 1, 19)

import torch
import torch.nn as nn

X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.3)
X_train_torch = torch.from_numpy(X_train).float()
X_test_torch = torch.from_numpy(X_test).float()
Y_train_torch = torch.from_numpy(Y_train).float()
Y_test_torch = torch.from_numpy(Y_test).float()

from torch.utils.data import TensorDataset, DataLoader
# 定义数据集
train_dataset = TensorDataset(X_train_torch, Y_train_torch)
# 定义数据加载器
train_data_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

#####

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=24, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 19)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# 初始化网络
model = Net()
# 选择优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 定义损失函数
criterion = nn.MSELoss()
# 训练模型
for epoch in range(100):
    for i, (features, labels) in enumerate(train_data_loader):
        features = features.permute(0,2,1)
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')

torch.save(model.state_dict(),"LSTM100.pth")



