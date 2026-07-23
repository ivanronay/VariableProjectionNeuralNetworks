# Basis for this code is taken from the following repository:
#   https://gitlab.com/AmonAttilaMiklos/1drgwvp/
#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com

# import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from VPExtra.models import *
from VPExtra.utility import *
from VPBase.models import *
from VPBase.data_generator import *
from VPBase.CWTLayer import *
from VPBase.trainer import *
from VPBase.utility import *
from statistics import mean, stdev
from sklearn.model_selection import KFold


def do_kfold(dset, n_splits, model_iterator, loss = torch.nn.BCELoss()):
    # Train/test indeces
    ### Define DataLoader Object ###
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    x = dset._samples
    y = dset._labels
    accuracies = []
    fold_rows = []

    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        print(f"Fold {fold + 1}:")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

        trainLoader = DataLoader(dset, batch_size=BS, sampler=train_subsampler)
        testLoader = DataLoader(dset, batch_size=BS, sampler=test_subsampler)

        model = next(model_iterator)
        optimizer = torch.optim.Adam(params = model.parameters(),lr=LR)

        tr_l, tr_a, te_l, te_a, se_1, pos_pred_1, se_0, pos_pred_0 = train(model, trainLoader, testLoader, EP, loss, optimizer, device,len(train_index), len(test_index))

        accuracies.append(max(te_a))
        plot_model_loss_acc(tr_l, tr_a, te_l, te_a,EP)
        row = log_fft_fold(
            "logs/training_folds.csv",
            dataset_name=file_name,
            model_name=type(model).__name__,
            train_size=len(train_index),
            test_size=len(test_index),
            learning_rate=LR,
            batch_size=BS,
            epochs=EP,
            hidden_layers=model.layer_params,
            train_loss=tr_l,
            train_accuracy=tr_a,
            test_loss=te_l,
            test_accuracy=te_a,
            sensitivity_1=se_1,
            positive_predictivity_1=pos_pred_1,
            sensitivity_0=se_0,
            positive_predictivity_0=pos_pred_0
        )
        fold_rows.append(row)
        print()
        if n_splits == 1:
            break
    
    print('\nList of possible accuracy:', accuracies)
    print('\nMaximum Accuracy That can be obtained from this model is:',
        max(accuracies)*100, '%')
    print('\nMinimum Accuracy:',
        min(accuracies)*100, '%')
    print('\nMean Accuracy:',
        mean(accuracies)*100, '%')
    print('\nStandard Deviation is:', stdev(accuracies))

    log_fft_summary(
        "logs/training_kfold_summary.csv",
        dataset_name=file_name,
        model_name=type(model).__name__,
        fold_rows=fold_rows
    )

def FFTSensorKFold(dset, n_splits=5):
    do_kfold(dset, n_splits, createFFT())

def createFFT():
    while True:
        yield FFTClassifier(signal_length, [20,20,20])

def MelSensorKFold(dset, n_splits=5):
    do_kfold(dset, n_splits, create_Mel())

def create_Mel():
    while True:
        for n_mels in [20, 40, 60, 80, 100]:
            model = MelClassifier(signal_length, [20,20,20], n_mels=n_mels)
            yield model

def CWT_SNNSensorKFold(dset, n_splits=5):
    # Training related parameters
    layer = 'VP' 
    mother_wavelet = ['RATGAUSS'] # 'MORLET' or 'RICKER' or 'RATGAUSS' or 'HERMITE' 
    input_length = dset._samples.shape[2]
    VP_PEN = 0.1
    VP_DIM = 3
    NR = [20,20,20]

    # RatGauss wavelet constants
    bmin = 0.01 # minimum absolute value of imaginary part of the poles of the rational term
    p = 3 # number of zeros of the polynomial term on the positive half-space
    r = 4 # number of the poles of the rational term
    a = -5.0
    b = 5.0

    frame_size = 100
    hop_size = 50

    vp_target = 2 # Target features. 0: coefficients, 1: approximation, 2: residual
    init_params = init_rgw_tire_sensor(p, r, VP_DIM,-1,1,device)
    dummy_model = create_model([layer,mother_wavelet],frame_size,VP_DIM,NR,VP_PEN,vp_target,a,b,p,r,bmin,init_params,device=device)
    
    do_kfold(dset, n_splits, createCWT_SNN(dummy_model.vp_params, frame_size, hop_size))

def createCWT_SNN(vp_params, frame_size, hop_size):
    while True:
        yield  CWTClassifierSNN([20,20,20],vp_params,frame_size,hop_size)

device= "cpu"
file_name='tire_sensor'
LR = 0.001
BS = 64
EP = 20

if __name__=="__main__":
    # Seed random generators
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load dataset
    # For CWT
    dset = MFASensorRevolutionsData(SHUFFLE=True, svm=False, add_dim=True) # add_dim if CWT
    signal_length = dset._samples.shape[2] # 2 if CWT 1 if not

    # For FFT and Mel
    # dset = MFASensorRevolutionsData(SHUFFLE=True, svm=False, add_dim=False) # add_dim if CWT
    # signal_length = dset._samples.shape[1] # 2 if CWT 1 if not


    # VPTireSensorTest()
    # VPKfoldTireSensortest()
    # VPKFoldGridSearch()
    # FFTSensorKFold(dset)
    # MelSensorKFold(dset)
    CWT_SNNSensorKFold(dset)