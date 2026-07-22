# Basis for this code is taken from the following repository:
#   https://gitlab.com/AmonAttilaMiklos/1drgwvp/
#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com

import time 
import csv
import numpy as np
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


def VPTireSensorTest():

    # Load dataset
    # Data
    file_name='tire_sensor'
    dset = MFASensorRevolutionsData(SHUFFLE=True, svm=False)

    # Training related parameters
    layer = 'VP' 
    mother_wavelet = ['RATGAUSS'] # 'MORLET' or 'RICKER' or 'RATGAUSS' or 'HERMITE' 
    N = len(dset)
    input_length = dset._samples.shape[2]
    LR = 0.0001
    BS = 32
    EP = 100
    VP_PEN = 5
    VP_DIM = 3
    NR = [5]

    # RatGauss wavelet constants
    bmin = 0.01 # minimum absolute value of imaginary part of the poles of the rational term
    p = 2 # number of zeros of the polynomial term on the positive half-space
    r = 5 # number of the poles of the rational term
    a = -1.0
    b = 1.0

    vp_target = 2 # Target features. 0: coefficients, 1: approximation, 2: residual
    init_params = init_rgw_tire_sensor(p, r, VP_DIM,-1,1,device)

    print(N,input_length)

    # Train/test indeces
    ### Define DataLoader Object ###
    M = int(0.8*N)
    tr_inds = torch.arange(M,device=device)
    te_inds = torch.arange(M, N,device=device)

    train_subsampler = torch.utils.data.SubsetRandomSampler(tr_inds)
    test_subsampler = torch.utils.data.SubsetRandomSampler(te_inds)

    trainLoader = DataLoader(dset, batch_size=BS, sampler=train_subsampler)
    testLoader = DataLoader(dset, batch_size=BS, sampler=test_subsampler)

    # Define model
    model = create_model([layer,mother_wavelet],input_length,VP_DIM,NR,VP_PEN,vp_target,a,b,p,r,bmin,init_params=init_params,device=device)
    model = model.to(device)

    ## Plotting ###
    plot_Psi(model,input_length,mother_wavelet,VP_DIM,p,r,bmin)

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

    write_to_log("training_log_tire.csv",file_name,tr_l, tr_a, te_l, te_a,NR,EP,BS,LR,VP_PEN,VP_DIM,a,b,p,r,bmin,layer,mother_wavelet,vp_target)
    with open("training_log_tire.csv", 'a', newline='') as csvfile:
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
    
    plot_Psi(model,input_length,mother_wavelet,VP_DIM,p,r,bmin)

    print("The time of execution of above program is :",
       (end-start), "s")

    print("Done")

def VPKfoldTireSensortest():

    device= "cpu"

    # Load dataset
    # Data
    file_name='tire_sensor'
    dset = MFASensorRevolutionsData(SHUFFLE=True, svm=False)

    # Training related parameters
    layer = 'VP' 
    mother_wavelet = ['RATGAUSS'] # 'MORLET' or 'RICKER' or 'RATGAUSS' or 'HERMITE' 
    N = len(dset)
    input_length = dset._samples.shape[2]
    LR = [0.01,0.0001]
    BS = 32
    EP = 200
    VP_PEN = 0.1
    VP_DIM = 3
    NR = [20,20,20]

    # RatGauss wavelet constants
    bmin = 0.01 # minimum absolute value of imaginary part of the poles of the rational term
    p = 3 # number of zeros of the polynomial term on the positive half-space
    r = 4 # number of the poles of the rational term
    a = -5.0
    b = 5.0

    vp_target = 2 # Target features. 0: coefficients, 1: approximation, 2: residual

    print(N,input_length)

    # init_params = init_rgw_tire_sensor(p, r, VP_DIM,-1,1,device)

    # Train/test indeces
    ### Define DataLoader Object ###
    skf = KFold(n_splits=5, shuffle=True, random_state=0)
    lst_accu_kfold = []

    x = dset._samples
    y = dset._labels

    for train_index, test_index in skf.split(x, y):

        init_params = init_rgw_tire_sensor(p, r, VP_DIM,-1,1,device)
        # init_params = None
        
        # Define model
        model = create_model([layer,mother_wavelet],input_length,VP_DIM,NR,VP_PEN,vp_target,a,b,p,r,bmin,init_params,device=device)
        loss = torch.nn.BCELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR[1])
        optimizer = torch.optim.Adam([{'params':model.vp_layers[0].weight, "lr": LR[0]},
                                      {'params':model.layers.parameters(), "lr": LR[1]}],
                                     lr=LR[1])

        good_inds = np.where(y == 0.0)[0]
        abnorm_inds = np.where(y == 1.0)[0]

        # Count the number of good and abnorm samples in the train set
        good_in_train = len([i for i in good_inds if i in train_index])
        abnorm_in_train = len([i for i in abnorm_inds if i in train_index])

        # Count the number of good and abnorm samples in the test set
        good_in_test = len([i for i in good_inds if i in test_index])
        abnorm_in_test = len([i for i in abnorm_inds if i in test_index])

        print("Train set:")
        print(f"Good samples: {good_in_train}")
        print(f"Abnorm samples: {abnorm_in_train}")

        print("Test set:")
        print(f"Good samples: {good_in_test}")
        print(f"Abnorm samples: {abnorm_in_test}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

        trainLoader = DataLoader(dset, batch_size=BS, sampler=train_subsampler)
        testLoader = DataLoader(dset, batch_size=BS, sampler=test_subsampler)

        ## Plotting ###
        plot_Psi(model,input_length,mother_wavelet,VP_DIM,p,r,bmin,a,b)

        # Train 
        tr_l, tr_a, te_l, te_a, se_1, pos_pred_1, se_0, pos_pred_0 = train(model, trainLoader, testLoader, EP, loss, optimizer, device,train_index.shape[0], test_index.shape[0])

        min_index = np.argmax(te_a)
        lst_accu_kfold.append(te_a[min_index])

        
        plot_Psi(model,input_length,mother_wavelet,VP_DIM,p,r,bmin,a,b)
        plot_model_loss_acc(tr_l, tr_a, te_l, te_a,EP)
        write_to_log("training_log_5fold_tire_sensor.csv",file_name,tr_l, tr_a, te_l, te_a,NR,EP,BS,LR,VP_PEN,VP_DIM,a,b,p,r,bmin,layer,mother_wavelet,vp_target,min_index)
    
            
    # Print the output.
    print('List of possible accuracy:', lst_accu_kfold)
    print('\nMaximum Accuracy That can be obtained from this model is:',
        max(lst_accu_kfold)*100, '%')
    print('\nMinimum Accuracy:',
        min(lst_accu_kfold)*100, '%')
    print('\nOverall Accuracy:',
        mean(lst_accu_kfold)*100, '%')
    print('\nStandard Deviation is:', stdev(lst_accu_kfold))

    with open("training_log_5fold_tire_sensor.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['List of possible accuracy', 'Maximum Accuracy','Minimum Accuracy', 'Overall Accuracy', 'Standard Deviation'])
        
        writer.writerow([
            ', '.join(map(str, lst_accu_kfold)),
            max(lst_accu_kfold),
            min(lst_accu_kfold),
            mean(lst_accu_kfold),
            stdev(lst_accu_kfold)
        ])
    
    
    print("Done")

def VPKFoldGridSearch():
    device= "cpu"

    # Training related parameters
    layer = 'VP' 
    mother_wavelet = ['RICKER'] # 'MORLET' or 'RICKER' or 'RATGAUSS' or 'HERMITE' 
    LR = [[0.01,0.0001],[0.001,0.0001],[0.01,0.001],[0.01,0.01]]
    BS = [32]
    EP = 200
    VP_PEN = [0.001,0.1,5]
    VP_DIM = 3
    NR = [20,20,20]

    # RatGauss wavelet constants
    bmin = 0.01 # minimum absolute value of imaginary part of the poles of the rational term
    p = 3 # number of zeros of the polynomial term on the positive half-space
    r = 4 # number of the poles of the rational term
    a = -5.0
    b = 5.0

    vp_target = 2 # Target features. 0: coefficients, 1: approximation, 2: residual

    db = 1
    for i in range(len(LR)):
        for j in range(len(VP_PEN)):
            for z in range(len(BS)):
                print(db,"/",len(LR)*len(VP_PEN)*len(BS))
                db= db + 1

                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)

                # Load dataset
                # Data
                file_name='tire_sensor'
                dset = MFASensorRevolutionsData(SHUFFLE=True, svm=False)

                N = len(dset)
                input_length = dset._samples.shape[2]

                # Train/test indeces
                ### Define DataLoader Object ###
                skf = KFold(n_splits=5, shuffle=True, random_state=0)
                lst_accu_kfold = []

                x = dset._samples
                y = dset._labels

                for train_index, test_index in skf.split(x, y):

                    init_params = None
                    # init_params = init_rgw_tire_sensor(p, r, VP_DIM,-1,1,device)
                    
                    # Define model
                    model = create_model([layer,mother_wavelet],input_length,VP_DIM,NR,VP_PEN[j],vp_target,a,b,p,r,bmin,init_params,device=device)
                    loss = torch.nn.BCELoss()
                    # optimizer = torch.optim.Adam(model.parameters(), lr=LR[1])
                    optimizer = torch.optim.Adam([{'params':model.vp_layers[0].weight, "lr": LR[i][0]},
                                                {'params':model.layers.parameters(), "lr": LR[i][1]}],
                                                lr=LR[i][1])

                    train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
                    test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

                    trainLoader = DataLoader(dset, batch_size=BS[z], sampler=train_subsampler)
                    testLoader = DataLoader(dset, batch_size=BS[z], sampler=test_subsampler)

                    # Train 
                    tr_l, tr_a, te_l, te_a, se_1, pos_pred_1, se_0, pos_pred_0 = train(model, trainLoader, testLoader, EP, loss, optimizer, device,train_index.shape[0], test_index.shape[0])

                    min_index = np.argmax(te_a)
                    lst_accu_kfold.append(te_a[min_index])

                    write_to_log("training_log_5fold_tire_sensor_grids.csv",file_name,tr_l, tr_a, te_l, te_a,NR,EP,BS[z],LR[i],VP_PEN[j],VP_DIM,a,b,p,r,bmin,layer,mother_wavelet,vp_target,min_index)

                with open("training_log_5fold_tire_sensor_grids.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write the header row
                    writer.writerow(['List of possible accuracy', 'Maximum Accuracy','Minimum Accuracy', 'Overall Accuracy', 'Standard Deviation'])
                    
                    # Write the data row
                    writer.writerow([
                        ', '.join(map(str, lst_accu_kfold)),
                        max(lst_accu_kfold) * 100,
                        min(lst_accu_kfold) * 100,
                        mean(lst_accu_kfold) * 100,
                        stdev(lst_accu_kfold)
                    ])
                
                
                print("Done")

def do_kfold(dset, n_splits, model_generator):
    # Train/test indeces
    ### Define DataLoader Object ###
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    accuracies = []
    fold_rows = []

    x = dset._samples
    y = dset._labels
    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        print(f"Fold {fold + 1}:")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

        trainLoader = DataLoader(dset, batch_size=BS, sampler=train_subsampler)
        testLoader = DataLoader(dset, batch_size=BS, sampler=test_subsampler)

        # model = MelClassifier(signal_length, [20,20,20], n_mels=n_mels[fold])
        model = next(model_generator())
        loss = torch.nn.BCELoss()
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
    do_kfold(dset, n_splits, lambda: FFTClassifier(signal_length, [20,20,20]))

def MelSensorKFold(dset, n_splits=5):
    do_kfold(dset, n_splits, create_Mel)

def create_Mel():
    for n_mels in [20, 40, 60, 80, 100]:
        model = MelClassifier(signal_length, [20,20,20], n_mels=n_mels)
        yield model

device= "cpu"
file_name='tire_sensor'
LR = 0.0001
BS = 32
EP = 200

if __name__=="__main__":
    # Seed random generators
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load dataset
    dset = MFASensorRevolutionsData(SHUFFLE=True, svm=False, add_dim=False)
    signal_length = dset._samples.shape[1]

    # VPTireSensorTest()
    # VPKfoldTireSensortest()
    # VPKFoldGridSearch()
    # FFTSensorKFold(dset)
    MelSensorKFold(dset)