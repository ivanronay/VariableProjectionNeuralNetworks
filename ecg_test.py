# Basis for this code is taken from the following repository:
#   https://gitlab.com/AmonAttilaMiklos/1drgwvp/
#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from VPBase.models import *
from VPBase.CWTLayer import *
from VPBase.data_generator import *
from VPBase.trainer import *
from VPBase.utility import *
import time 
import csv


def VPCWTECGTest():
    # Seed random generators
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device= "cpu"

    # Load dataset
    # Data
    file_name='unbalanced_heartbeat'
    dset = ECGData(SHUFFLE=True, ISSVM=False,dataset=file_name,device=device)

    # Training related parameters
    layer = 'VP' 
    mother_wavelet = ['RATGAUSS'] # 'MORLET' or 'RICKER' or 'RATGAUSS' or 'HERMITE' 
    N = len(dset)
    M = dset._training_data_size
    input_length = dset._samples.shape[2]
    LR = 0.01
    BS = 512
    EP = 23
    VP_PEN = 0.01
    VP_DIM = 10
    NR = 9

    # RatGauss wavelet constants
    bmin = 0.1 # minimum absolute value of imaginary part of the poles of the rational term
    p = 3 # number of zeros of the polynomial term on the positive half-space
    r = 4 # number of the poles of the rational term
    a = -1.0
    b = 1.0

    vp_target = 0 # Target features. 0: coefficients, 1: approximation, 2: residual

    print(N,input_length)

    # Train/test indeces
    ### Define DataLoader Object ###
    tr_inds = torch.arange(M,device=device)
    te_inds = torch.arange(M, N,device=device)

    train_subsampler = torch.utils.data.SubsetRandomSampler(tr_inds)
    test_subsampler = torch.utils.data.SubsetRandomSampler(te_inds)

    trainLoader = DataLoader(dset, batch_size=BS, sampler=train_subsampler)
    testLoader = DataLoader(dset, batch_size=BS, sampler=test_subsampler)

    # Define model
    model = create_model([layer,mother_wavelet],input_length,VP_DIM,[NR],VP_PEN,vp_target,a,b,p,r,bmin,init_params=None,device=device)
    model = model.to(device)

    ## Plotting ###
    plot_Psi(model,input_length,mother_wavelet,VP_DIM,p,r,bmin,a,b)

    loss = torch.nn.BCELoss()
    # loss = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = time.time()

    # Train 
    tr_l, tr_a, te_l, te_a, se_1, pos_pred_1, se_0, pos_pred_0 = train(model, trainLoader, testLoader, EP, loss, optimizer, device,tr_inds.shape[0], te_inds.shape[0])
    
    end = time.time()
    
    print( se_1[-1], pos_pred_1[-1], se_0[-1], pos_pred_0[-1])

    plot_model_loss_acc(tr_l, tr_a, te_l, te_a,EP)

    write_to_log("training_log.csv",file_name,tr_l, tr_a, te_l, te_a,NR,EP,BS,LR,VP_PEN,VP_DIM,a,b,p,r,bmin,layer,mother_wavelet,vp_target)
    with open("training_log.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['sensitivity/precision (Se) - label_1', 'positivepredictivity/recall (+P) - label_1','sensitivity/precision (Se) - label_0', 'positivepredictivity/recall (+P) - label_0'])
        
        # Write the data row
        writer.writerow([
            se_1[-1] * 100,
            pos_pred_1[-1] * 100,
            se_0[-1] * 100,
            pos_pred_0[-1] * 100
        ])
    
    plot_Psi(model,input_length,mother_wavelet,VP_DIM,p,r,bmin,a,b)

    print("The time of execution of above program is :",
       (end-start), "s")

    print("Done")

if __name__=="__main__":
    VPCWTECGTest()