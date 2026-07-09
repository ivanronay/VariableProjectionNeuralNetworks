function [psi, t] = rgwwavf(LB,UB,N,WNAME)
    global n
    global p
    global alpha
    global bmin
    global sigma

    t = linspace(LB,UB,N);
    psi = psi_fun(t, alpha(1:2:2*n), alpha(2:2:2*n), alpha(2*n+1:2*n+p), bmin, sigma);
    psi = psi/norm(psi);
end

