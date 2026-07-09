#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com
##  Last modified: 28.05.2024

import torch
import torch.nn as nn
from VPBase.CWTLayer import *


class VPCWTNN(nn.Module):
    """Continuous wavelet transform net, with improved approximation of wavelet coefficients.
    The first layer of the network is a so-called Variable Projection (VP) layer. It projects the input
    x, onto the subspace spanned by the dilated and translated wavelet functions -> wavelet coeffs are
    approximated as the the coordinates of this projection in the subspace.
    """
    def __init__(self, wavegenfun, nparams, input_length, vp_latent_dim,vp_target,p,r,b_min,a,b,neuron_n=[5], penalty=None, init_vp=None, device=None):
        super().__init__()
        n_out = 2
        self.vp_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.vpl_number = 1

        self.add_vp_layer(vp_layer(wavegenfun, n_in=input_length, p=p,r=r,b_min=b_min,a=a,b=b,target=vp_target,
                            n_out=vp_latent_dim, nparams=nparams,
                            device=device, penalty=penalty, init=init_vp))

        if vp_target == 0:
            n0 = vp_latent_dim
        else: 
            n0 = input_length
        
        self.bnorm = nn.BatchNorm1d(n0, affine=True)

        for n in neuron_n:
            self.add_layer(nn.Linear(n0, n))
            self.add_layer(nn.ReLU())
            n0 = n
        if 2 == n_out:
            self.add_layer(nn.Linear(n0, 1)) 
            self.add_layer(nn.Sigmoid())
        else:
            self.add_layer(nn.Linear(n0, n_out))
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def add_vp_layer(self, layer):
        self.vp_layers.append(layer)

    def test_mode(self):
        for i in range(self.vpl_number):
            psi, _, _ = self.vp_layers[i].ada(self.vp_layers[i].weight)
            self.vp_layers[i].Phi = psi
            self.vp_layers[i].Phip = torch.linalg.pinv(psi)
        
    def train_mode(self):
        for i in range(self.vpl_number):
            self.vp_layers[i].Phi = None
            self.vp_layers[i].Phip = None

    def forward(self, x):
        vplayer = self.vp_layers[0]
        x = vplayer(x)
        x = torch.squeeze(x)
        if 1 == len(x.shape): x = x.unsqueeze(0) # BS = 1 

        # me = x.mean(dim=1, keepdim=True)
        # se = x.std(dim=1, keepdim=True)
        # x = (x - me)/se

        # x = self.bnorm(x)

        for layer in self.layers:
            x = layer(x)
        
        return x
