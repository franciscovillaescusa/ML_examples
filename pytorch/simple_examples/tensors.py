import torch
import numpy as np

# define a tensor
t = torch.tensor([1,2,3])                   #way-1
a = np.arange(10);  t = torch.tensor(a)     #way-2 same dtype as numpy
a = np.arange(10);  t = torch.Tensor(a)     #way-3 transformed to float32 tensor
a = np.arange(10);  t = torch.as_tensor(a)  #way-4 same dtype as numpy
a = np.arange(10);  t = torch.from_numpy(a) #way-5 same dtype as numpy
a = np.arange(10);  t = torch.tensor(a, dtype=torch.float32) #way-6 explicit dtype
# options 2 and 3 will create a copy of a, while for options 4,5 a and t share
# the same underlying data. E.g. a[0]=0 will change a but also t in 4,5. Option 4
# is slightly better as 5 only applies to numpy arrays

# tensor attributes
t.shape;  t.size()  #are the same
t.dtype
t.device
t.layout
t.numel()     #number of total elements in a tensor. Useful for reshaping

# some useful functions
t = torch.eye(2)     #[[1.,0.],[0.,1.]]
t = torch.zeros(2,2)
t = torch.ones(2,2)
t = torch.rand(2,2)

# move the tensor to the GPU
t = t.cuda()                    
device = torch.device('cuda:1') #move to the second GPU

# move a tensor to numpy array
t.numpy()

# reshape/stack 
t.reshape(2,5)
t.reshape(1,-1) #for the second dimension pytorch will figure out the correct number
t.reshape(-1);  t.squeeze();  t.flatten();  t.view(t.numel())  #create a 1D tensor
t.flatten(start_dim=1);  #flatten only from first dimension
t = torch.stack((t1,t2,t3))

# Pytorch only supports operations between same data type tensors (float,int...)

# images are represented in Pytorch as [batch, color, height, width]

# this will work
t1 = torch.tensor([[1,1],[1,1]], dtype=torch.float32)
t2 = torch.tensor([2,4], dtype=torch.float32)
t1 + t2
# t2 is broadcasted to the shape of t1. To see what it is doing use this
np.broadcast_to(t2.numpy(), t1.shape)

# conditional operations (0-False, 1-True)
t = torch.tensor([[1,2,3],[-1,2,0],[3,-8,7]], dtype=torch.float32)
t.eq(0)  #where the tensor is equal to 0
t.ge(0)  #where the tensor is equal or greater than 0
t.gt(0)  #where the tensor is greater than 0
t.lt(0)  #where the tensor is less than 
t.le(0)  #where the tensor is equal or less than 0

# other operations
t.abs()
t.sqrt()
t.neg()  #return the negative values of the tensor
t.sum();  t.sum(dim=0)
t.prod() #product of all elements
t.mean()
t.std()
t.max();  t.max(dim=0)
t.argmax() #gives the index of the maximum value in t
t.t() #transpose of a tensor

# get the value of a tensor
t.mean().item()

# random numbers
seed = 1
torch.manual_seed(seed)



#################################################################################################
#################################################################################################

# define a tensor as part of graph
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# check if a tensor has gradients
a.requires_grad
