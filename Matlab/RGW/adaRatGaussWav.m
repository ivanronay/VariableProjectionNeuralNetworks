function [Phi,dPhi,Ind] = adaRatGaussWav(alpha, t, n, p, r, bmin)
% p: number of zeros in P
% r : number of poles in R
% n : number of wavelet coefficients
% bmin: minimum absolute value of imaginary part of R's poles
% alpha: (p1, ..., pp, r0real, r0imag, ..., rrreal, rrimag, s1, x1, s2, x2, ..., sn, xn, sigma)

% Some useful constants for indexing
rootend = p;
polebeg = p+1; poleend = p+1+2*r-1;
wavebeg = p+1+2*r;
L = 2+2*r+p+1;
N = length(t);

% Initialize Phi, dPhi and Ind
Phi = zeros(N, n);
dPhi = zeros(N, 2*n+(p+2*r+1));
Ind = zeros(2, 2*n+(p+2*r+1));

% Generate the wavelets and derivatives w.r.t. alpha
for k=0:n-1

    % Break up alpha to make the code readable
    begindzers = k*L+1;
    endindzers = begindzers+p-1;
    pk = alpha(1:p);
    
    begindpoles = k*L+polebeg;
    endindpoles = begindpoles+2*r-1;
    ak = alpha(polebeg:2:poleend-1);
    betak = alpha(polebeg+1:2:poleend);

    begindsig = k*L+poleend+3; 
    endindsig = begindsig;
    sigma = alpha(end);

    % Current dilation and translation
    s = alpha(wavebeg+2*k);
    x = alpha(wavebeg+2*k+1);
    tt = (t-x)/s;

    % Generate the next wavelet and associated derivatives
    [Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma] = psi_fun(tt, ak, betak, pk, bmin, sigma);

    % Transpose everything that needs to be transposed + save functions
    Psi = (Psi/sqrt(s)).';
    Phi(:,k+1) = real(Psi);
    dPsix = real(dPsix).'/sqrt(s); dPsia = real(dPsia).'/sqrt(s); dPsib = real(dPsib).'/sqrt(s);
    dPsip = real(dPsip).'/sqrt(s); dPsiSigma = real(dPsiSigma).'/sqrt(s);

    % Save derivatives and Ind values
    
    % zers
    dPhi(:,begindzers:endindzers) = dPsip;
    Ind(1,begindzers:endindzers) = k+1;
    Ind(2,begindzers:endindzers) = 1:p;

    % poles
    dPhi(:, begindpoles:2:endindpoles-1) = dPsia;
    Ind(1, begindpoles:2:endindpoles-1) = k+1;
    Ind(2, begindpoles:2:endindpoles-1) = polebeg:2:poleend-1;

    dPhi(:, begindpoles+1:2:endindpoles) = dPsib;
    Ind(1, begindpoles+1:2:endindpoles) = k+1;
    Ind(2, begindpoles+1:2:endindpoles) = polebeg+1:2:poleend;

    % Wavelet parameters
    dPsis = -0.5*s.^(-3/2).*Psi + dPsix.*(-1.*s.^(-2)).*(t-x).';
    dPsit = dPsix.*(-1)/s;

    begindwav = k*L+wavebeg;
    endindwav = begindwav + 1;
    dPhi(:, begindwav:endindwav) = [dPsis, dPsit];
    Ind(1, begindwav:endindwav) = k+1;
    Ind(2, begindwav:endindwav) = wavebeg + 2*k:wavebeg + 2*k+1;

    % Sigma
    dPhi(:, begindsig:endindsig) = dPsiSigma;
    Ind(1, begindsig:endindsig) = k+1;
    Ind(2, begindsig:endindsig) = length(alpha);
end



end