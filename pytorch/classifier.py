import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
from pylab import *
import sys,os


# define the dataset class
class Data_set(data.Dataset):
    def __init__(self,points=1000):
        self.pos = torch.rand((points,2))
        self.label = torch.zeros(points, dtype=torch.int64)
        d = ((self.pos[:,0]-0.5)**2 + self.pos[:,1]**2).sqrt()
        indexes = np.where(d.numpy()>0.7)[0]
        self.label[indexes]=1
        self.len = points
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        return self.pos[index], self.label[index]

# define the arquitecture
class Model(nn.Module):
    def __init__(self,hidden1,hidden2):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(2, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)
    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))
        

    

points     = 1000
num_epochs = 1000
batch_size = 50
hidden1    = 20
hidden2    = 20

# define the dataset and the training set
dataset  = Data_set(points)
training = data.DataLoader(dataset=dataset, batch_size=batch_size,
                           shuffle=True, num_workers=1)
pos_tot   = dataset.pos.numpy()
label_tot = dataset.label.numpy()

# define the coordinates of the validation set
pos_test = torch.zeros((100*100,2), dtype=torch.float32)
x,y = torch.meshgrid([torch.linspace(0,1,100), torch.linspace(0,1,100)])
x = x.contiguous().view(1,-1);  y = y.contiguous().view(1,-1)
pos_test[:,0] = x;  pos_test[:,1] = y


# define the network, loss and optimizer
net = Model(hidden1, hidden2)
loss_func = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.01)

# plot original results
plt.ion()
fig=figure(figsize=(11*1.5,3.3*1.5))     #give dimensions to the figure
ax1=plt.subplot(1,3,1);  ax1.set_aspect('equal')
ax2=plt.subplot(1,3,2);  ax2.set_aspect('equal')
ax3=plt.subplot(1,3,3)
ax3.set_yscale('log')
ax1.scatter(pos_tot[:,0], pos_tot[:,1], c=label_tot)

# do a loop over all the epochs
total_loss = np.zeros(num_epochs)
epochs     = np.arange(num_epochs)
for epoch in xrange(num_epochs):

    # do a loop over the different batches
    for pos,label in training:

        label_pred = net(pos.view(-1,2))
        loss = loss_func(label_pred, label)
        optimizer.zero_grad()   #set gradients to 0
        loss.backward()         #make backpropagation
        optimizer.step()        #update weights
        total_loss[epoch] += loss.item()

    if epoch%10==0:
        ax2.cla() #clear axes
        _, predicted = torch.max(net(pos_test), 1)
        cax = ax2.imshow(predicted.view(100,100).t().numpy(),origin='lower',
                         extent=[0, 1, 0, 1])
        #ax2.scatter(pos_test[:,0], pos_test[:,1], c=predicted)
        ax3.plot(epochs[:epoch], total_loss[:epoch], 'b-', lw=1)
        plt.pause(0.0001)        


plt.pause(10)
