import numpy as np
import sys,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt


# define a dataset class
class Data_set(data.Dataset):
    def __init__(self, slope, bias, points=100, gaussian_noise_amplitude=1.0):
        n = torch.distributions.Normal(0.0, 1.0) #define a normal distribution generator
        self.x = torch.linspace(0,10,points, dtype=torch.float32)
        self.y = slope*self.x + bias + gaussian_noise_amplitude*n.sample((points,))
        self.len = self.x.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.x[index], self.y[index]


# define the neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  #nn.Module.__init__(self)
        self.fc = nn.Linear(1, 1)
        
    def forward(self,x):
        return self.fc(x)

    
torch.manual_seed(7)

# variables
slope = 5.0
bias  = -5.0
points = 300
batch_size = 20
num_epochs = 1000

# define network, loss and optimizer
net       = Model()
loss_func = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# define the dataset and the training set
dataset    = Data_set(slope,bias,points)
train_data = data.DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=True, num_workers=1)

# get the whole x- and y- data for plotting
x_data = dataset.x
y_data = dataset.y

# do a loop over many epochs
plt.ion() #interacting plotting
total_loss = np.zeros(num_epochs)
epochs     = np.arange(num_epochs)
for epoch in xrange(num_epochs):

    # do a loop over the different batches in a given epoch
    for x,y in train_data:

        y_pred = net(x.view(-1,1))
        loss = loss_func(y_pred, y.view(-1,1)) #the order is important: y_pred, y
        optimizer.zero_grad()   #set gradients to 0
        loss.backward()         #make backpropagation
        optimizer.step()        #update weights
        total_loss[epoch] += loss.item() 
        
    # plot and show learning process
    plt.subplot(1,2,1)
    plt.cla() #clear axes
    plt.scatter(x_data.numpy(), y_data.numpy())
    plt.plot(x_data.numpy(), net(x_data.view(-1,1)).detach().numpy(), 'r-', lw=1)
    plt.text(0.5, 0, 'w=%.2f : b=%.2f'%(net.fc.weight.detach().numpy(), net.fc.bias.detach().numpy()),
             fontdict={'size': 12, 'color':  'red'})
    plt.subplot(1,2,2)
    plt.cla() #clear axes
    plt.yscale('log')
    plt.plot(epochs[:epoch], total_loss[:epoch], 'b-', lw=1)
    plt.pause(0.0001)        
    
plt.ioff()
plt.show()



# if there are 100 elements in total, and batch=35.
# The sizes of the batches will be 45,45 and 10
for x,y in train_data:
    print len(x),len(y)


################ How to list the parameters ##################
# With this we can see the value of the weights in the arquitecture
params = list(net.parameters())
# params[0] is the weights of the linear layer
# params[1] is the bias of the linear layer
# or we can see the parameters of each layer as:
# w = net.fc.weight
# b = net.fc.bias

#for param in net.parameters():
#    print param, param.grad
    
