import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import numpy as np
import scipy.special as si
import sys,os
import matplotlib.pyplot as plt


# data loader
class Dataset(data.Dataset):
    def __init__(self):
        x = np.linspace(0.0,2.0,300, dtype=np.float32)
        y = np.zeros(300, dtype=np.float32)
        indexes = np.where(x<1.0)[0]
        y[indexes] = np.sin((x[indexes]-1.0)*30.0)
        indexes = np.where((x>=1.0) & (x<1.5))[0]
        y[indexes] = (x[indexes])**2-1.0
        indexes = np.where((x>=1.5))[0]
        y[indexes] = si.jv(0,(x[indexes]-1.5)*40)*2.0 -0.75

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.x[index], torch.tensor([self.x[index], self.x[index]**2]), self.y[index]
        

# neural network arquitecture
class Model1(nn.Module):
    def __init__(self, hidden1, hidden2, hidden3, hidden4, hidden5):
        super(Model1,self).__init__()
        self.fc1 = nn.Linear(1,       hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.fc5 = nn.Linear(hidden4, hidden5)
        self.fc6 = nn.Linear(hidden5, 1)

    def forward(self,x):
        x1 = F.leaky_relu(self.fc1(x),  negative_slope=0.01)
        x2 = F.leaky_relu(self.fc2(x1), negative_slope=0.01)
        x3 = F.leaky_relu(self.fc3(x2), negative_slope=0.01)
        x4 = F.leaky_relu(self.fc4(x3), negative_slope=0.01)
        x5 = F.leaky_relu(self.fc5(x4), negative_slope=0.01)
        x6 = self.fc6(x5)
        return x6

# neural network arquitecture
class Model2(nn.Module):
    def __init__(self, hidden1, hidden2, hidden3, hidden4, hidden5):
        super(Model2,self).__init__()
        self.fc1 = nn.Linear(2,       hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.fc5 = nn.Linear(hidden4, hidden5)
        self.fc6 = nn.Linear(hidden5, 1)

    def forward(self,x):
        x1 = F.leaky_relu(self.fc1(x),  negative_slope=0.01)
        x2 = F.leaky_relu(self.fc2(x1), negative_slope=0.01)
        x3 = F.leaky_relu(self.fc3(x2), negative_slope=0.01)
        x4 = F.leaky_relu(self.fc4(x3), negative_slope=0.01)
        x5 = F.leaky_relu(self.fc5(x4), negative_slope=0.01)
        x6 = self.fc6(x5)
        return x6

# set the random seed
torch.manual_seed(9) #9 is among the best
    
# define the neural network
hidden1, hidden2, hidden3, hidden4, hidden5 = 50, 60, 70, 60, 50
model1 = Model1(hidden1, hidden2, hidden3, hidden4, hidden5)
model2 = Model2(hidden1, hidden2, hidden3, hidden4, hidden5)

# define the optimizer
optimizer1 = optim.Adam(model1.parameters(), lr=0.002)
optimizer2 = optim.Adam(model2.parameters(), lr=0.002)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# define the data loader
dataset = Dataset()
train_data = data.DataLoader(dataset=dataset, batch_size=60, shuffle=True, num_workers=0)

# define the loss
loss_func = nn.MSELoss()


plt.ion()

x_plot1 = torch.linspace(0,2,1000)
x_plot2      = np.zeros((1000,2), dtype=np.float32)
x_plot2[:,0] = np.linspace(0,2,1000)
x_plot2[:,1] = np.linspace(0,2,1000)**2
x_plot2      = torch.tensor(x_plot2)


plt.axis((0,2,-1.9,1.4))
plt.scatter(dataset.x.numpy(), dataset.y.numpy(), c='r')
c = input('?')

for epoch in xrange(1000):
    total_loss1, total_loss2 = 0, 0
    for x,x2,y in train_data:
        
        y_pred1 = model1(x.view(-1,1))
        loss1 = loss_func(y_pred1, y.view(-1,1))
        total_loss1 += loss1
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        y_pred2 = model2(x2)
        loss2 = loss_func(y_pred2, y.view(-1,1))
        total_loss2 += loss2
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    print 'Epoch = %03d ----> losses = %.4e %.4e'%(epoch,total_loss1,total_loss2)
    
    y_pred1 = model1(x_plot1.view(-1,1))
    y_pred2 = model2(x_plot2)
    
    plt.cla()
    plt.axis((0,2,-1.9,1.4))
    plt.scatter(dataset.x.numpy(), dataset.y.numpy(), c='r')
    plt.plot(x_plot1.numpy(), y_pred1.detach().numpy(), c='b', lw=1)
    #plt.plot(x_plot1.numpy(), y_pred2.detach().numpy(), c='b', lw=1, linestyle='--')
    plt.text(0.1, -1.6, 'Epoch %03d : losses %.3e %.3e'%(epoch,total_loss1,total_loss2),
             fontdict={'size': 12, 'color':  'green'})
    plt.pause(0.01)


plt.ioff()
plt.show()
