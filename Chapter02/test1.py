import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNeuralNet(nn.Module):
    def __init__(self, input_size, n_nodes, output_size):
        super(MyNeuralNet, self).__init__()
        self.operationOne = nn.Linear(input_size, n_nodes)
        self.operationTwo = nn.Linear(n_nodes, output_size)
    def forward(self, x):
        x = F.relu(self.operationOne(x))
        x = self.operationTwo(x)
        x = F.sigmoid(x)
        return x

#create torch nn
my_network = MyNeuralNet(input_size = 3, n_nodes = 2, output_size = 1)        


#data load numpy array
import numpy as np
admit_data = np.genfromtxt('Chapter02/admit_status.csv', delimiter = ',', skip_header = 1)
# print(admit_data)


#make torch data from numpy
admit_tensor = torch.from_numpy(admit_data)
# print(admit_tensor)


#make dataset   train, test  x input y valid
x_train = admit_tensor[:300, 1:].float()
y_train = admit_tensor[:300, 0].float()
x_test = admit_tensor[300:, 1:].float()
y_test = admit_tensor[300:, 0].float()

# (판단)기준  활성함수
# binary cross-entropy loss
criterion = nn.BCELoss()

optimizer = torch.optim.SGD(my_network.parameters(), lr=0.01)


for epoch in range(100):
    y_pred = my_network(x_train)
    loss_score = criterion(y_pred, y_train)
    print('epoch: ', 'y_pred : ', epoch, y_pred, 'loss: ', loss_score)
    
    optimizer.zero_grad()
    loss_score.backward()
    optimizer.step()