#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com
##  Last modified: 28.05.2024

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as P

def genfun_morlet(m, n, params, p,r,dtype=torch.float, device=None):

    # Define base interval as the effective support of the non dilated and non translated Morlet wavelet
    t = torch.linspace(-4, 4, m)

    # Define the mother wavele
    morl = lambda t: torch.exp(-t**2/2) * torch.cos(5*t)
    dmorl = lambda t: -t*torch.exp(-t**2/2)*torch.cos(5*t) - torch.exp(-t**2/2)*torch.sin(5*t)*5

    psi = torch.zeros(m, n)
    dpsi = torch.zeros(m, 2*n)
    ind = torch.zeros(2, 2*n)

    # This implementation of Morlet wavelets only optimizes dilation and translation parameters -> all of nparams are 
    # of this type in a (\lambda_1, \tau_1, \lambda_2, \tau_2, ....) manner
    for k in range(n):
        pars = params[2*k:2*k+2] # Current lambda_k, tau_k parameters
        if pars[0] < 0:
            print("WARNING: one of the scale param become smaller than 0")
            print("Old value: ",pars[0]) 
            pars[0] = 0.3
            print("New value: ",pars[0])
        psi[:,k] = 1/np.sqrt(pars[0]) * morl((t - pars[1])/pars[0])  # *(t[1]-t[0])
        dpsi[:,2*k] = -0.5*pars[0]**(-3/2)*morl((t - pars[1])/pars[0]) - pars[0]**(-5/2)*(t - pars[1])*dmorl((t - pars[1])/pars[0]) # *(t[1]-t[0])
        dpsi[:,2*k+1] = -pars[0]**(-3/2)*dmorl((t - pars[1])/pars[0]) # *(t[1]-t[0])
        ind[0, 2*k] = int(k) # kth basic function
        ind[1, 2*k] = int(2*k) # 2*k lambda_k
        ind[0, 2*k+1] = int(k) # kth basic function
        ind[1, 2*k+1] = int(2*k+1) # params 2*k+1 tau_k 
    
    ind = ind.to(torch.int64)

    return psi, dpsi, ind

def genfun_ricker(m, n, params,p,r,b_min,a,b, dtype=torch.float, device=None):

    alpha = params.tolist()
    # Define base interval as the effective support of the non dilated and non translated Ricker wavelet
    t = torch.linspace(-5, 5, m,device=device)

    # Define the mother wavele
    c = 2/(math.sqrt(3)*math.sqrt(math.sqrt(math.pi)))
    rick = lambda t: c*torch.exp(-t**2/2) * (1-t**2)
    drick = lambda t: c*(-t*torch.exp(-t**2/2)* (1-t**2)-torch.exp(-t**2/2)*2*t)
    
    psi = torch.zeros(m, n,device=device)
    dpsi = torch.zeros(m, 2*n,device=device)
    ind = torch.zeros(2, 2*n,device=device)

    for k in range(n):
        pars = alpha[2*k:2*k+2] # Current lambda_k, tau_k parameters
        s = pars[0]
        if s < 0 : s = s**2
        x = pars[1]
        psi[:,k] = 1/math.sqrt(s) * rick((t - x)/s) 
        dpsi[:,2*k] = -0.5*s**(-3/2)*rick((t - x)/s) - s**(-5/2)*(t - x)*drick((t - x)/s) # dPsi/dLambda
        dpsi[:,2*k+1] = -s**(-3/2)*drick((t - x)/s)
        ind[0, 2*k] = int(k) # kth basic function
        ind[1, 2*k] = int(2*k) # 2*k lambda_k
        ind[0, 2*k+1] = int(k) # kth basic function
        ind[1, 2*k+1] = int(2*k+1) # params 2*k+1 tau_k 

    ind = ind.to(torch.int64)
    return psi, dpsi, ind


'''
beta : scalar - imag part of one of the poles of R
bmin: scalar - minimum absolute value of imaginary part of R's poles
'''
def b_k(beta, bmin):
    b_k = beta**2 + bmin
    db_k = 2*beta # derivative of b_k
    return b_k, db_k

'''
x : vector - dilated, translated datapoints of the effective support of the function
a : scalar - real part one of the poles of R
beta : scalar - imag part one of the poles of R
bmin: scalar - minimum absolute value of imaginary part of R's poles
'''
def Q(x, a, beta, bmin):
    b, db = b_k(beta, bmin)
    Qf = x**4 + x**2*(2*b**2 - 2*a**2) + a**4 + 2*a**2*b**2 + b**4
    dQx = 4*x**3 + 2*x*(2*b**2 - 2*a**2)
    dQa = -4*x**2*a + 4*a**3 + 4*a*b**2
    dQb = db*(4*x**2*b + 4*b**3 + 4*a**2*b)
    return Qf, dQx, dQa, dQb

'''
x : vector - dilated, translated datapoints of the effective support of the function
a : scalar - real part one of the poles of R
beta : scalar - imag part one of the poles of R
bmin: scalar - minimum absolute value of imaginary part of R's poles
'''
def R(x, a, beta, bmin):
    Qf, dQx, dQa, dQb = Q(x, a, beta, bmin)
    Rf = Qf**(-1)
    dRx = -Qf**(-2)*dQx
    dRa = -Qf**(-2)*dQa
    dRb = -Qf**(-2)*dQb
    return Rf, dRx, dRa, dRb

'''
x : dilated, translated datapoints of the effective support of the function
ak : real part of the poles of R
betak : imag part of the poles of R
pk : zeros of the polynom on the positive half-space
bmin: minimum absolute value of imaginary part of R's poles
sigma: parameter of the Gaussian function
'''
def psi_fun(x, ak, betak, pk, bmin, sigma,device):
    n = len(ak) # number of poles of the rational term R
    m = len(pk) # number of zeros of the polynomial term P

    Rfun = torch.ones(len(x),device=device)
    r_k = torch.zeros(n, len(x),device=device)
    dRx_k = torch.zeros(n, len(x),device=device)
    dRa_k = torch.zeros(n, len(x),device=device)
    dRb_k = torch.zeros(n, len(x),device=device)

    # Construct the polynomial term
    pk = np.array(pk)
    zeros = np.concatenate((pk, -pk, [0]))

    # old np version, numpy.polynomial is recommended
    # Palg = np.poly(zeros)
    # Pf = np.polyval(Palg, x)
    # dPx = np.polyval(np.polyder(Palg), x)

    Palg = P.polyfromroots(zeros)
    Pf = P.polyval(x,Palg)
    dPx = P.polyval(x,P.polyder(Palg))

    # Construct the rational term R
    for k in range(n):
        r, rx, ra, rb = R(x, ak[k], betak[k], bmin)
        Rfun = Rfun*r # multiply the n number of elementary weight modifier polynomials 
        r_k[k,:] = r
        dRx_k[k,:] = rx
        dRa_k[k,:] = ra
        dRb_k[k,:] = rb

    # Construct R's derivative w.r.t. x
    dRx = torch.zeros(len(x),device=device)
    for k in range(n):
        rr = torch.cat((r_k[:k], r_k[k+1:])) # erase kth row because of chain rule of derivatives
        dRx += dRx_k[k, :] * torch.prod(rr, dim=0)

    # Construct the mother wavelet and derivatives
    Psi = Pf*Rfun*torch.exp(-x**2/sigma**2)

    # Derivatives w.r.t.x
    dPsix = dPx*Rfun*torch.exp(-x**2/sigma**2) + Pf*dRx*torch.exp(-x**2/sigma**2) - 2*x/sigma**2 * Psi

    # Derivatives w.r.t. a, b
    dPsia = torch.zeros(n, len(x),device=device)
    dPsib = torch.zeros(n, len(x),device=device)
    
    for k in range(n):
        rr = torch.cat((r_k[:k], r_k[k+1:])) # erase kth row because of partial derivatives
        dPsia[k,:] = dRa_k[k,:]*torch.prod(rr,dim=0)*Pf*torch.exp(-x**2/sigma**2)
        dPsib[k,:] = dRb_k[k,:]*torch.prod(rr,dim=0)*Pf*torch.exp(-x**2/sigma**2)

    # Derivatives w.r.t. p
    dPsip = torch.zeros(m, len(x),device=device)
    for k in range(m):
        dPp = -( Pf/( (x - pk[k])*(x + pk[k]) ) )*2*pk[k]
        roots = np.copy(pk)
        roots = np.delete(roots, k)
        roots = np.concatenate((roots, -roots, [0]))
        Pcurr = P.polyfromroots(roots)
        Pf = P.polyval(x, Pcurr)

        dPp = -Pf*2*pk[k]

        dPsip[k, :] = dPp*Rfun*torch.exp(-x**2/sigma**2)


    # Derivatives w.r.t. sigma
    dPsiSigma = Psi*2*x**2*sigma**(-3)

    return Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma


'''
p : number of zeros in P
r : number of poles in R
n : number of wavelet coefficients
bmin : minimum absolute value of imaginary part of R's poles
alpha : (p1, ..., pp, r0real, r0imag, ..., rrreal, rrimag, s1, x1, s2, x2, ..., sn, xn, sigma)
m : length of the input vector
'''
def adaRatGaussWav(m, n, params, p, r,bmin,a,b, smin=0.01, s_square= False,dtype=torch.float, device=None):

    alpha = params.tolist()
    # Some useful constants for indexing
    polebeg = p # p+1 in matlab, because start index is 1, not 0
    poleend = p+1+2*r-1 # ok, because np.array[0:3] eq array(1:3) in matlab
    wavebeg = p+1+2*r-1 # p+1+2*r in matlab, because start index is 1, not 0

    L = 2+2*r+p+1 # number of params of a dilated, translated wavelet

    # N = len(t)
    N = m
    # Define base interval as the effective support of the non dilated and non translated wavelet
    t = torch.linspace(a, b, m,device=device) 

    # Initialize Phi, dPhi and Ind
    Phi = torch.zeros(N, n,device=device)
    dPhi = torch.zeros(N, n*L,device=device)
    Ind = torch.zeros(2, n*L,device=device)

    # common parameters for all dilated, translated wavelets
    sigma = alpha[-1]
    ak = alpha[polebeg:poleend-1:2] # only real part of poles
    betak = alpha[polebeg+1:poleend:2] # only imag part of poles
    pk = alpha[0:p]

    # Generate the wavelets and derivatives w.r.t. alpha
    for k in range(n):
        # Break up alpha to make the code readable
        begindzers = k*L # k*L+1 in matlab, because start index is 1, not 0
        endindzers = begindzers+p # p-1 in matlab
        
        begindpoles = k*L+polebeg
        endindpoles = begindpoles+2*r # 2*r-1 in matlab

        begindsig = k*L+poleend+2 # dil trans miatt

        # Current dilation and translation
        s = alpha[wavebeg+2*k]
        ss = s
        if s < 0: ss = s**2 + smin
        if s_square: ss = s**2 + smin
        x = alpha[wavebeg+2*k+1]
        tt = (t-x)/ss

        # Generate the next wavelet and associated derivatives
        [Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma] = psi_fun(tt, ak, betak, pk, bmin, sigma,device)

        # Transpose everything that needs to be transposed + save functions
        Psi = Psi/math.sqrt(ss)
        Phi[:,k] = Psi.real
        dPsix = dPsix.real/math.sqrt(ss)
        dPsia = torch.transpose(dPsia.real,0,1)/math.sqrt(ss)
        dPsib = torch.transpose(dPsib.real,0,1)/math.sqrt(ss)
        dPsip = torch.transpose(dPsip.real,0,1)/math.sqrt(ss)
        dPsiSigma = dPsiSigma.real/math.sqrt(ss)


        # Save derivatives and Ind values
        
        # zeros
        dPhi[:,begindzers:endindzers] = dPsip
        Ind[0,begindzers:endindzers] = k 
        Ind[1,begindzers:endindzers] = torch.arange(0, p)

        # poles
        dPhi[:, begindpoles:endindpoles-1:2] = dPsia
        Ind[0, begindpoles:endindpoles-1:2] = k 
        Ind[1, begindpoles:endindpoles-1:2] = torch.arange(polebeg, poleend-1, step=2)

        dPhi[:, begindpoles+1:endindpoles:2] = dPsib
        Ind[0, begindpoles+1:endindpoles:2] = k
        Ind[1, begindpoles+1:endindpoles:2] = torch.arange(polebeg+1, poleend, step=2)

        # Wavelet parameters
        dPsis = -0.5*ss**(-3/2)*Psi + dPsix*(-1*ss**(-2))*(t-x).t() # .t() pythonban nem biztos, hogy kell
        if s_square: dPsis = dPsis * 2*s
        dPsit = dPsix*(-1)/ss

        begindwav = k*L+wavebeg
        endindwav = begindwav + 2 # 1 in matlab
        dPsis = torch.unsqueeze(dPsis, 1)
        dPsit = torch.unsqueeze(dPsit, 1)
        dPhi[:, begindwav:endindwav] = torch.cat((dPsis, dPsit), dim=1)
        Ind[0, begindwav:endindwav] = k
        Ind[1, begindwav:endindwav] = torch.arange(wavebeg + 2*k, wavebeg + 2*k+2)

        # Sigma
        dPhi[:, begindsig] = dPsiSigma
        Ind[0, begindsig] = k
        Ind[1, begindsig] = len(alpha) - 1

        Ind = Ind.to(torch.int64)

    return Phi,dPhi,Ind

def hermite_ada(m, n, params,p ,r,b_min,a,b, dtype=torch.float, device=None):
    alpha = params.tolist()
    dilation, translation = alpha
    t = torch.linspace(-5, 5, m)
    x = dilation * (t - translation)
    w = torch.exp(-0.5 * x ** 2)
    dw = -x * w
    pi_sqrt = np.sqrt(np.sqrt(np.pi))
    # Phi, dPhi
    Phi = torch.zeros(m, n)
    Phi[:, 0] = 1
    Phi[:, 1] = 2 * x
    for j in range(1, n - 1):
        Phi[:, j + 1] = 2 * (x * Phi[:, j] - j * Phi[:, j - 1])
    Phi[:, 0] = w * Phi[:, 0] / pi_sqrt
    dPhi = torch.zeros(m, 2 * n)
    dPhi[:, 0] = dw / pi_sqrt
    dPhi[:, 1] = dPhi[:, 0]
    f = 1.0
    for j in range(1, n):
        f *= j
        Phi[:, j] = w * Phi[:, j] / torch.sqrt(torch.tensor(2 ** j * f)) / pi_sqrt
        dPhi[:, 2 * j] = torch.sqrt(torch.tensor(2 * j)) * Phi[:, j - 1] - x * Phi[:, j]
        dPhi[:, 2 * j + 1] = dPhi[:, 2 * j]
    
    t = t.unsqueeze(1)

    dPhi[:, 0::2] = dPhi[:, 0::2] * (t - translation)
    dPhi[:, 1::2] = -dPhi[:, 1::2] * dilation
    # ind
    ind = torch.zeros(2, 2*n)
    ind[0, 0::2] = torch.arange(n, dtype=torch.int64)
    ind[0, 1::2] = torch.arange(n, dtype=torch.int64)
    ind[1, 0::2] = torch.zeros(1, n)
    ind[1, 1::2] = torch.zeros(1, n)
    ind = ind.to(torch.int64)
    return Phi, dPhi, ind