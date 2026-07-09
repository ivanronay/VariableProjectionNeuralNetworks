import csv
import numpy as np
import random
import torch
from VPBase.data_generator import *
from VPBase.wavelets import *
import datetime
from torch.utils.data import DataLoader
from VPBase.models import *
import copy

date = datetime.datetime.now()
date_str = date.strftime("%Y-%m-%d %H:%M")

def plot_model_loss_acc(tr_l, tr_a, te_l, te_a,epoch):
    # Plotting Loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(tr_l, label='Training Loss', color='blue')
    plt.plot(te_l, label='Test Loss', color='orange')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xlim(0,epoch)
    plt.legend()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(tr_a, label='Training Accuracy', color='green')
    plt.plot(te_a, label='Test Accuracy', color='red')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xlim(0,epoch)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_Psi(model,input_length,mother_wavelet,VP_DIM,p,r,b_min,a,b):

    param = model.vp_layers[0].weight
        
    if 'MORLET' == mother_wavelet[0] :
            psi, _, _ = genfun_morlet(input_length,VP_DIM,param,p,r)
    elif 'RICKER' == mother_wavelet[0] :
            psi, _, _ = genfun_ricker(input_length,VP_DIM,param,p,r,b_min,a,b)
    elif 'RATGAUSS' == mother_wavelet[0] :
            psi, _, _ = adaRatGaussWav(input_length,VP_DIM,param,p,r,b_min,a,b)
    elif 'HERMITE' == mother_wavelet[0] :
            psi, _, _ = hermite_ada(input_length,VP_DIM,param,p ,r,b_min)

    # Plotting the functions
    time = np.linspace(a, b, len(psi[:, 0]))
    plt.figure(figsize=(10, 6))
    for j in range(psi.shape[1]):  # Iterate over columns (functions) of psi matrix
        # Adding text with translation and dilation parameters
        # text = f'Translation: {translations[i]:.2f}\nDilation: {dilations[i]:.2f}'
        # plt.plot(time,psi[:, j]) # label=f'Function {i+1}, '+text 
        plt.plot(psi[:, j]) # label=f'Function {i+1}, '+text 

    plt.title('Functions represented by Psi matrix', fontsize=18)
    # plt.title('Morlet wavelet', fontsize=18)
    # plt.xlabel('Data Points',fontsize=14)
    plt.ylabel('Function Values',fontsize=14)
    plt.xlabel('Effective support of the mother wavelet',fontsize=14)
    # plt.xlim(-1, 1)
    # plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.xlim(0, input_length)
    plt.show()
    print(p)
    if 'RATGAUSS' == mother_wavelet[0]:
        c = -1.5
        d = 1.5
        alpha = torch.zeros(p+2*r+2+1)
        alpha[:p+2*r] = param[:p+2*r]
        alpha[p+2*r] = 1.0
        alpha[p+2*r+1] = 0.0
        alpha[-1] = param[-1]
        psi, _, _ = adaRatGaussWav(input_length,1,alpha,p,r,b_min,c,d)
        time = np.linspace(c, d, len(psi[:, 0]))
        plt.figure(figsize=(10, 6))
        plt.plot(time,psi[:, 0],color="red",linewidth=2)
        plt.title('Mother Wavelet',fontsize=18)
        # plt.xlabel('Data Points',fontsize=14)
        plt.xlabel('Effective support of the mother wavelet',fontsize=14)
        plt.xlim(c, d)
        plt.ylabel('Function Values',fontsize=14)
        plt.grid(True)
        plt.show()
    
    param = param.tolist()
    param = [round(p, 2) for p in param]
    print("params: ",param)

def write_to_log(filename, dataset_name, tr_l, tr_a,te_l, te_a,nr,ep,bs,lr,vp_pen,vp_dim,a,b,p,r,bmin,layer,mother_wavelet,vp_target,index=-1):
     
    target_str = ["coefficients","approximation","residual"]
    learning_params = "lr: "+str(lr)+", bs: "+str(bs) +", epoch: "+str(ep)
    learning_params += ", vp_pen: " + str(vp_pen) + ", vp_dim: " + str(vp_dim) + " neuron: "+str(nr)+", " + " index of best epoch: "+str(index)+", "
    learning_params += target_str[vp_target]
    learning_params += " (" + str(a) + "," + str(b) + ") "
    learning_params += "zeros,poles,bmin: " + str(p) + "," + str(r) + "," + str(bmin) + ", "
    tr_l_str = "{:.4f}".format(tr_l[index])
    tr_a_str = "{:.2f}".format(tr_a[index]*100)
    te_l_str = "{:.4f}".format(te_l[index])
    te_a_str = "{:.2f}".format(te_a[index]*100)
    # Open the log file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the data to the CSV file
        writer.writerow([date_str, dataset_name,layer,mother_wavelet, learning_params,tr_l_str, te_l_str,tr_a_str , te_a_str])


def init_ratgauss(pz, rp, vp_dim,a,b,device):
    alpha0 = torch.rand(pz + 2 * rp,device=device)
    alpha1 = torch.rand(2 * vp_dim,device=device)
    alpha1[::2] = 0.3 + alpha1[::2]  # init dilatation
    alpha1[1::2] = a + 2 * b* alpha1[1::2]  # init translation
    # [0, 1, 10, -1.5, 1.5, 0.01, 0.01]
    # return torch.tensor([0.056405529379844666, 0.11612903326749802, 0.6572902798652649,0.05, 0.7793733477592468,-0.1, 0.9664774537086487,0.1, 0.019999999552965164])

    # return torch.tensor([0.25999999046325684, 0.1194470077753067, 0.6572902798652649,0.2, 0.7793733477592468,-0.2, 0.9664774537086487,0.3, 0.18847742676734924])
    # return torch.tensor([0.8761,    2.3669,    1.2115,    1.1703,    0.0664,    1.7979,   -0.1031,    1.5966,   1.0765,    1.0367,    0.9658,    1.8529, 2.3238, 0.9585, 2.1238, 0.8585,2.2238, 0.7585,  -1.2285])
    return torch.cat((alpha0, alpha1, torch.tensor([0.3],device=device)))

def init_rgw_tire_sensor(pz, rp, vp_dim,a,b,device):
    alpha0 = torch.rand(pz + 2 * rp,device=device)
    alpha1 = torch.rand(2 * vp_dim,device=device)
    alpha1[::2] = 0.3 + alpha1[::2]  # init dilatation
    alpha1[1::2] = a + 2 * b* alpha1[1::2]  # init translation
    return torch.cat((alpha0, alpha1, torch.tensor([0.3],device=device)))

def init_dil_trans(nparams,a,b,device):
    init = torch.rand(nparams,device=device)
    init[1::2] = a + 2*b*init[1::2] # init translations
    return init

def init_dil_trans_hermite():
    return torch.tensor([1.0,0.0]) # dilation and translation param of Hermite system

def create_model(condition,input_length,vp_dim,nr,penalty,vp_target,a,b,p,r,b_min,init_params,device):

    configurations = {
        'VP': {
            'MORLET': (lambda: init_dil_trans(vp_dim*2,a,b), lambda init: 
                       VPCWTNN(genfun_morlet, vp_dim*2, input_length, vp_dim,vp_target,p,r,b_min,a,b, neuron_n=nr, penalty=penalty, init_vp=init)),
            'RICKER': (lambda: init_dil_trans(vp_dim*2,a,b,device), lambda init: 
                       VPCWTNN(genfun_ricker, vp_dim*2, input_length, vp_dim,vp_target,p,r,b_min,a,b, neuron_n=nr, penalty=penalty, init_vp=init)),
            'RATGAUSS': (lambda: init_ratgauss(p, r, vp_dim,a,b,device), lambda init: 
                         VPCWTNN(adaRatGaussWav, vp_dim*2+2*r+p+1, input_length, vp_dim,vp_target,p,r,b_min,a,b, neuron_n=nr, penalty=penalty, init_vp=init,device=device)),
            'HERMITE': (lambda: init_dil_trans_hermite(), lambda init: 
                       VPCWTNN(hermite_ada, 2, input_length, vp_dim,vp_target,p,r,b_min,a,b, neuron_n=nr, penalty=penalty, init_vp=init))
        }
    }

    type_key, wavelet_key = condition
    init_fn, model_fn = configurations.get(type_key).get(wavelet_key[0])
    if None == init_params:
        init = init_fn()
    else:
         init_params = copy.deepcopy(init_params)
         init = init_params
    model = model_fn(init)

    return model

def plot_approx_signals_ecg():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dset = ECGData(SHUFFLE=True, ISSVM=False)

    labels = dset._labels
    label_0_indices = np.where(labels == 0)[0][500:510]
    label_1_indices = np.where(labels == 1)[0][500:510]
    indices = np.concatenate(([550,5751],label_0_indices, label_1_indices))

 
    VP_DIM = 10
    input_length = dset._samples.shape[2]
    params = [0.8446453417979894, 0.9566453742216224, 5.536812464641108e-27,
-4.6643611690391834e-18, 0.24101270405131597, 1.023011545676396, 0.33802734531726236, 0.9804493213887779, 0.18940176537790732, 0.44936965860543576, 0.2641220643016294,
 1.3889746811350878, -0.9214035455382484,
 0.11095606976667599, -0.2684619609996233,
 1.128070504202262, 0.5897152346413169,
 1.4447057166426553, -0.561836718935027,
 0.8583797732564299, 0.35937859969923325,
 1.0154386321476498, -0.3074163105436838,
 0.9805006700309169, -0.01293235220479621,
 0.992316844392078, 1.1893629949815294,
 0.4649982324469111, -0.5071014660895201,
 0.18447338153502602, -0.2287873452952801, 
0.3809042447443462]
    phi, dpsi, ind = adaRatGaussWav(input_length,VP_DIM,params)
    phip = torch.linalg.pinv(phi)

    plt.figure(figsize=(4, 5))
    for i in range(len(indices)):
        x = torch.tensor(dset._samples[indices[i]])
        x = x.float()
        x = torch.transpose(x, 0, 1)
        coeffs = phip @ x
        y_est = phi @ coeffs
        
        plt.subplot(2, 1, 1)
        time = np.linspace(0, 1, len(x))
        plt.plot(time, x)  
        # plt.plot(time,y_est)

        print("Index (tr): ", indices[i])
        print("Label (norm/abnorm): ", dset._labels[indices[i]])

        # plt.title('Input and the approx. signal (Index (tr): ' + str(indices[i]) + ', Label (norm/abnorm): '
        #          + str(dset._labels[indices[i]]) + ')')
        # plt.title('Input and the approx. signal')
        plt.title('Input signal')
        # plt.xlabel('Data Points')
        plt.xlabel('Time in seconds (s)')
        plt.ylabel('Function Values')
        # plt.legend()
        plt.xlim(0, 1)
        plt.grid(True)
        
        '''
        plt.subplot(2, 1, 1)
        for i in range(phi.shape[1]): 
            plt.plot(phi[:, i])

        plt.title('Functions represented by Psi matrix')
        plt.xlabel('Data Points')
        plt.ylabel('Function Values')
        # plt.legend()
        plt.xlim(0, input_length)
        plt.grid(True)
        '''
        plt.subplot(2, 1, 2)
        # plt.plot(x)
        # x_values = [-0.92, -0.268, 0.5897, -0.5618, 0.3593, -0.30741, -0.012, 1.189, -0.5071, -0.2287]
        x_values_ind = []
        for i in range(phi.shape[1]): 
            max_value_pos = np.argmax(phi[:, i])
            x_values_ind.append(max_value_pos)
        t = torch.linspace(-1, 1, input_length)
        x_values = [t[i] for i in x_values_ind]
        print(x_values)
        print([-0.92, -0.268, 0.5897, -0.5618, 0.3593, -0.30741, -0.012, 1.189, -0.5071, -0.2287])
        y_values = coeffs
        for i in range(phi.shape[1]):
            plt.bar(x_values[i], y_values[i], width=0.1)
        # plt.bar(x_values, y_values, width=0.1)
        plt.axhline(0, color='black', linewidth=1)

        # plt.title('Coeff of the ECG signal')
        plt.xlabel('Effective support of the mother wavelet')
        plt.ylabel('Coeffs. values')
        # plt.legend()
        plt.xlim(-1, 1)
        plt.grid(False)

        

        plt.show()


if __name__=="__main__":
    # plot_approx_signals_ecg()
    print("Done")


