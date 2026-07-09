# Basis for this code is taken from the following repository:
#   https://gitlab.com/AmonAttilaMiklos/1drgwvp/
#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com

import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from sklearn import preprocessing

from sklearn.utils import shuffle

    
def LoadECGData(SHUFFLE=True, ISSVM=False,dataset='balanced_qrs',device=None):
    
    if 'unbalanced_heartbeat' == dataset:    
        train_data = np.genfromtxt('data/ecg_unbalanced_ds1_training_01.csv', delimiter=',')
        test_data = np.genfromtxt('data/ecg_unbalanced_ds2_testing_01.csv', delimiter=',')
        training_data_size = len(train_data)
    
    if SHUFFLE:
        np.random.seed(2)
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

    data = np.concatenate((train_data, test_data), axis=0)

   
    samples = data[:,0:-1]
    labels = data[:,-1]

    if ISSVM:
        zerinds = np.where(labels == 0)
        labels[zerinds] = -1

    samples = torch.tensor(samples,device=device,dtype=torch.float32)
    labels = torch.tensor(labels,device=device,dtype=torch.float32)

    return samples, labels, training_data_size
    
class ECGData(Dataset):
    def __init__(self, SHUFFLE=True, ADD_DIM=True, ISSVM=False,dataset='balanced_qrs',device=None):
        
        # Load the data
        self._samples, self._labels, self._training_data_size = LoadECGData(SHUFFLE, ISSVM,dataset,device)
        self._add_dim = ADD_DIM
        if ADD_DIM:
            self._samples = self._samples.unsqueeze(1)

    def __len__(self):
        return self._samples.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self._add_dim:
            sample = self._samples[index,:,:]
        else:
            sample = self._samples[index, :]
        label = self._labels[index]
        return sample, label
    


'''
Load tire revolution dataset
'''
def LoadTireRevolutions(MEAS_PATH='data/tire_sens/abnormal_tire_revolutions.mat', FIELD_NAME='abnorm_meas', SPLIT_RATIO=0.8, SHUFFLE=True):
    
    data_mat = sio.loadmat(MEAS_PATH)
    data = np.array(data_mat[FIELD_NAME])

    return TrainTestSplit(data, SPLIT_RATIO, SHUFFLE)

def TrainTestSplit(MEAS_SET, SPLIT_RATIO, SHUFFLE=True):
    if SHUFFLE:
        np.random.shuffle(MEAS_SET)
    samples = MEAS_SET[:,0:-1]
    pad = np.zeros((1,6))
    samples = ZeroPadRows(samples, pad, pad)

    labels = MEAS_SET[:,-1]

    n = labels.shape[0]
    tr_x = samples[0:int(np.floor(n*SPLIT_RATIO))]
    te_x = samples[int(np.floor(n*SPLIT_RATIO)):]

    tr_y = labels[0:int(np.floor(n*SPLIT_RATIO))]
    te_y = labels[int(np.floor(n*SPLIT_RATIO)):]

    return tr_x, te_x, tr_y, te_y, samples, labels

def ZeroPadRows(x, lpad, rpad):

    N = x.shape[0]
    px = np.zeros((N, x.shape[1] + lpad.shape[1] + rpad.shape[1]))
    for k in range(N):
        row = np.expand_dims(x[k], axis=0)
        px[k,:] = np.hstack((lpad, row, rpad))

    return px

'''
Create pytorch dataset from loaded tire revolutions
'''
class MFASensorRevolutionsData(Dataset):
    def __init__(self, MEAS_PATH='data/abnormal_tire_revolutions.mat', FIELD_NAME='abnorm_meas', SPLIT_RATIO=1.0, SHUFFLE=True, add_dim=True, svm=False):
        
        # Load the data
        _, _, _, _, self._samples, self._labels = LoadTireRevolutions(MEAS_PATH, FIELD_NAME, SPLIT_RATIO, SHUFFLE)

        '''
        for i in range(len(self._samples)):
            print("label: ", self._labels[i] - 1)
            print("index: ", i)
            plt.figure(figsize=(16, 4))
            plt.plot(self._samples[i,:],color="Red",linewidth=2)
            plt.title('Tire sensor signal',fontsize=18)
            # plt.xlabel('Data Points',fontsize=14)
            plt.xlabel('Data points',fontsize=14)
            plt.xlim(0, 512)
            plt.ylabel('Voltage (mV)',fontsize=14)
            plt.grid(True)
            plt.show()
        '''
        

        self._add_dim = add_dim
        if add_dim:
            self._samples = np.expand_dims(self._samples, axis=1)
        self._labels = self._labels - 1 # Pytorch CrossEntropy expects 0s or 1s not 1s and 2s
        if svm: # For SVM classification we change the labels to -1 and 1
            zerinds = np.where(self._labels == 0)
            self._labels[zerinds] = -1

    def __len__(self):
        return self._samples.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self._add_dim:
            sample = self._samples[index,:,:]
        else:
            sample = self._samples[index, :]
        label = self._labels[index]
        return sample.astype(np.float32), label.astype(np.float32)
