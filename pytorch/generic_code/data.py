import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time


# This routine reads all bottlenecks
def read_all_bottlenecks(root_in, realizations):

    # check if file already exists
    fout = 'all_bottlenecks.npy'
    if os.path.exists(fout):  return np.load(fout)

    # define the matrix containing all bottlenecks
    bn = np.zeros((realizations*153,512), dtype=np.float32)

    # do a loop over all realizations and save results to file
    count = 0
    for i in range(realizations):
        data = np.load('%s/%d/image_ALL_XYZ_z.npy'%(root_in,i))
        elements = data.shape[0]
        bn[count:count+elements] = data
        count += elements
    print('Found %d bottlenecks'%count)
    np.save(fout, bn)
    return bn

# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, realizations, root_in):

        # get the size and offset depending on the type of dataset
        if   mode=='train':  
            size, offset = int(realizations*153*0.70), int(realizations*153*0.00)
        elif mode=='valid':  
            size, offset = int(realizations*153*0.15), int(realizations*153*0.70)
        elif mode=='test':   
            size, offset = int(realizations*153*0.15), int(realizations*153*0.85)
        elif mode=='all':
            size, offset = int(realizations*153*1.00), int(realizations*153*0.00)
        else:    raise Exception('Wrong name!')

        # define size
        self.size = size

        # read the value of the parameters and normalize them
        params_orig = np.loadtxt('cosmo_params.txt', unpack=False)
        min_params  = np.min(params_orig, axis=0)
        max_params  = np.max(params_orig, axis=0)
        params_orig = (params_orig - min_params)/(max_params - min_params)
        params = np.zeros((realizations*153,5), dtype=np.float32)
        for i in range(realizations*153):
            params[i] = params_orig[i//153]

        # read the bottlenecks
        bn = read_all_bottlenecks(root_in, realizations)

        # randomly shuffle the cubes. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(realizations*153) #only shuffle realizations, not rotations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # get the corresponding bottlenecks and parameters
        self.input  = torch.tensor(bn[indexes],     dtype=torch.float32)
        self.output = torch.tensor(params[indexes], dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
def create_dataset(mode, seed, realizations, root_in, batch_size):
    data_set = make_dataset(mode, seed, realizations, root_in)
    dataset_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return dataset_loader
