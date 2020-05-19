import numpy as np
import torch
import sys,os
sys.path.append('../')
import data as data
import architecture

#################################### INPUT ##########################################
# data parameters
root_bn = '/mnt/ceph/users/fvillaescusa/CDS_2019/bottleneck_regression/bn_512_mean_new'
root_Pk = '/mnt/ceph/users/fvillaescusa/CDS_2019/bottleneck_regression/ps'
seed    = 1
realizations = 2000

# architecture parameters
h1 = 1000
h2 = 400
h3 = 100
dropout_rate = 0.0

# training parameters
batch_size = 10

minimum = np.array([0.1, 0.03, 0.5, 0.8, 0.6])
width   = np.array([0.4, 0.04, 0.4, 0.4, 0.4])

# name of output files
name   = '3hd_250_250_250_0.0_3e-4'
fout   = 'results/%s.txt'%name
fmodel = 'models/%s.pt'%name
#####################################################################################

# get GPU if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print("CUDA Not Available")
    device = torch.device('cpu')

# get the test dataset
test_loader  = data.create_dataset('test', seed, realizations, root_bn, root_Pk, 
                                   batch_size)

# get the number of elements in the test set
size = 0
for x, y in test_loader:
    size += x.shape[0]

# define the array with the results
pred = np.zeros((size,5), dtype=np.float32)
true = np.zeros((size,5), dtype=np.float32)

# get the parameters of the trained model
#model = architecture.model_1hl(bins_SFRH, h1, 6, dropout_rate)
#model = architecture.model_2hl(bins_SFRH, h1, h2, 6, dropout_rate)
model = architecture.model_3hl(512+158, h1, h2, h3, 5, dropout_rate)
model.load_state_dict(torch.load(fmodel))
model.to(device=device)

# loop over the different batches and get the prediction
offset = 0
model.eval()
for x, y in test_loader:
    with torch.no_grad():
        x    = x.to(device)
        y    = y.to(device)
        y_NN = model(x)
        length = x.shape[0]
        pred[offset:offset+length] = y_NN.cpu().numpy()
        true[offset:offset+length] = y.cpu().numpy()
        offset += length

# compute the rmse; de-normalize
error_norm = ((pred - true))**2
pred  = pred*width + minimum
true  = true*width + minimum
error = (pred - true)**2

print('Error^2 norm      = %.3e'%np.mean(error_norm))
print('Error             = %.3e'%np.sqrt(np.mean(error)))
print('Relative error Om = %.3e'%np.sqrt(np.mean(error[:,0])))
print('Relative error Ob = %.3e'%np.sqrt(np.mean(error[:,1])))
print('Relative error h  = %.3e'%np.sqrt(np.mean(error[:,2])))
print('Relative error ns = %.3e'%np.sqrt(np.mean(error[:,3])))
print('Relative error s8 = %.3e'%np.sqrt(np.mean(error[:,4])))

# save results to file
results = np.zeros((size,10))
results[:,0:5]  = true
results[:,5:10] = pred
np.savetxt(fout, results)

