# imports
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

# import matplotlib.pyplot as plt
import os
from multiprocessing import Process, freeze_support

os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_data = 1000
num_epoch = 5000

noise = init.normal(torch.FloatTensor(num_data,1),std=1)
x = init.uniform(torch.Tensor(num_data,1),-10,10)

# y = 2*x+3
y = -x**3 - 8*(x**2) + 3 

y_noise = y + noise
 
# model = nn.Linear(1,1)
# output = model(Variable(x))

model = nn.Sequential(
            nn.Linear(1,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1),
        ).cuda()

# loss_func = nn.MSELoss()
loss_func = nn.L1Loss()

optimizer = optim.SGD(model.parameters(),lr=0.01)

writer = SummaryWriter('TAcademy/runs/step2')

# train
loss_arr =[]
label = Variable(y_noise.cuda())


for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(Variable(x.cuda()))
    
    loss = loss_func(output,label)
    
    # if i % 10 == 0:
    #     # print(loss)
    #     param_list = list(model.parameters())
    #     # print(param_list[0].data,param_list[1].data)
    #     # print('pre ',param_list[0].data.numpy(),param_list[1].data.numpy())

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
    # print(loss)    
        writer.add_scalar('training loss',                    
                        loss.data.cpu().data.numpy(), #GPU to cpu
                        i
                        )

        for name, param in model.named_parameters():
            # writer.add_histogram('histogram', param.clone().cpu().data.numpy(), n_iter)
            writer.add_histogram('histogram', param.clone().cpu().data.numpy())
#

    # print(i, loss )

    # loss_arr.append(loss.data.numpy()[0])
# writer.close()

param_list = list(model.parameters())
print(param_list[0].data,param_list[1].data)
