# Basis for this code is taken from the following repository:
#   https://gitlab.com/AmonAttilaMiklos/1drgwvp/
#   (C) Ámon Attila Miklós
#       Eötvös Loránd University,
#       Department of Numerical Analysis
#       E-mail: aattila2000@gmail.com

import torch
import torch.nn as nn
from torch.autograd.function import Function

class vp_layer(nn.Module):
    """Basic Variable Projection (VP) layer class.
    The output of a single VP operator is forwarded to the subsequent layers.

        Input
        ----------
        ada: callable
            Builder for the function system and its derivatives (see e.g., 'ada_hermite').
        n_in: int
            Input dimension of the VP layer.
        n_out: int
            Output dimension of the VP layer.
        nparams: int
            Number of trainable weights,
            e.g., nparams=2 in the case of 'ada_hermite' function.
        penalty: L2 regularization penalty that is added directily to the training loss (see e.g., 'vpfun').
            This can be intepreted as a skip connection from this layer to the cost function. Default: 0.0.
        device: torch.device. Default: None.
            The desired device of the returned tensor(s).
        init: a list of values to initialize the VP layer.
            Default for Hermite functions: init=[0.1, 0.0].
        """

    def __init__(self, ada, n_in, n_out, nparams, p,r,b_min,a,b, penalty=0.0, target=2, dtype=torch.float, device=None, init=None):
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.target = target
        self.penalty = penalty
        self.Phi = None
        self.Phip = None
        self.ada = lambda params: ada(n_in, n_out, params, p ,r,b_min,a,b, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(init)

    def forward(self, input):
        return vpfun.apply(input, self.weight, self.ada, self.device, self.penalty,self.target,self.Phi,self.Phip)

class vpfun(Function):
    """Performs orthogonal projection, i.e. projects the input 'x' to the
    space spanned by the columns of 'Phi', where the matrix 'Phi' is provided by the 'ada' function.

    Input
    ----------
    x: torch Tensor of size (N,C,L) where
        N is the batch_size,
        C is the number of channels, and
        L is the number of signal samples
    params: torch Tensor of floats
          Contains the nonlinear parameters of the function system stored in Phi.
          For instance, if Phi(params) is provided by 'ada_hermite',
          then 'params' is a tensor of size (2,) that contains the dilation and the translation
          parameters of the Hermite functions.
    ada: callable
        Builder for the function system. For a given set of parameters 'params',
        it computes the matrix Phi(params) and its derivatives dPhi(params).
        For instance, in this package 'ada = ada_hermite' could be used.
    device: torch.device
             The desired device of the returned tensor(s).
    penalty: L2 regularization penalty that is added to the training loss.
              For instance, in the case of classification, the training loss is calculated as

                  loss = cross-entropy loss + penalty * ||x - projected_input||_2 / ||x||_2,

              where the projected_input is equal to the orthogonal projection of
              the 'x' to the columnspace of 'Phi(params)',
              i.e., projected_input =  Phi.mm( torch.linalg.pinv(Phi(params).mm(x) )

    Output
    -------
    coeffs: torch Tensor
             Coefficients of the projected input signal:

                 projected_input =  Phi.mm( torch.linalg.pinv(Phi(params).mm(x) ),

             where coeffs = torch.linalg.pinv(Phi(params).mm(x)
    """

    @staticmethod
    def forward(ctx, x, params, ada, device, penalty,target,Phi,Phip):
        ctx.device = device
        ctx.penalty = penalty
        ctx.target = target
        dphi = None
        ind = None
        phi = None
        phip = None
        if Phi == None:
            phi, dphi, ind = ada(params)
            phip = torch.linalg.pinv(phi)
        else:
            phi = Phi
            phip = Phip
        coeffs = phip @ torch.transpose(x, 1, 2)
        y_est = torch.transpose(phi @ coeffs, 1, 2)
        nparams = torch.tensor(max(params.shape))
        ctx.save_for_backward(x, phi, phip, dphi, ind, coeffs, y_est, nparams)

        if target == 1:
            return y_est
        elif target == 2:
            return x - y_est
        else: # self.target == 0
            return coeffs

    @staticmethod
    def backward(ctx, dy):
        x, phi, phip, dphi, ind, coeffs, y_est, nparams = ctx.saved_tensors
        #dx = dy @ phip
        if ctx.target == 0:
            dx = torch.squeeze(dy) @ phip
        else:
            dx = (torch.squeeze(dy) @ phi) @ phip
            if ctx.target == 2:
                dx = torch.squeeze(dy) - dx
        dp = None
        wdphi_r = (x - y_est) @ dphi
        phipc = torch.transpose(phip, -1, -2) @ coeffs  # (N,L,C)

        batch = x.shape[0]
        t2 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        jac1 = torch.zeros(
            batch, 1, phi.shape[0], nparams, dtype=x.dtype, device=ctx.device)
        jac3 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        for j in range(nparams):
            rng = ind[1, :] == j
            indrows = ind[0, rng]
            jac1[:, :, :, j] = torch.transpose(dphi[:, rng] @ coeffs[:, indrows, :], 1, 2)  # (N,C,L)
            t2[:, :, indrows, j] = wdphi_r[:, :, rng]
            jac3[:, :, indrows, j] = torch.transpose(phipc, 1, 2) @ dphi[:, rng]

        # Jacobian matrix of the forward pass with respect to the nonlinear parameters 'params'
        if ctx.target == 0:
            jac = -phip @ jac1 + phip @ (torch.transpose(phip, -1, -2) @ t2) + jac3 - phip @ (phi @ jac3)
        else:
            jac = jac1 - phi @ (phip @ jac1) + torch.transpose(phip, -1, -2) @ t2
            if ctx.target == 2:
                jac = -jac
        dy = dy.unsqueeze(-1)
        res = (x - y_est) / (x ** 2).sum(dim=2, keepdim=True)
        res = res.unsqueeze(-1)
        dp = (jac * dy).mean(dim=0).sum(dim=1) - 2 * \
            ctx.penalty * (jac1 * res).mean(dim=0).sum(dim=1)

        return dx, dp, None, None, None, None, None, None