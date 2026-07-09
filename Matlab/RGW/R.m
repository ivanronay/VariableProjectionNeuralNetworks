function [R, dRx, dRa, dRb] = R(x, a, beta, bmin)
    [Qf, dQx, dQa, dQb] = Q(x, a, beta, bmin);
    R = Qf.^(-1);
    dRx = -Qf.^(-2).*dQx;
    dRa = -Qf.^(-2).*dQa;
    dRb = -Qf.^(-2).*dQb;
end

