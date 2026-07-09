function [Q, dQx, dQa, dQb] = Q(x, a, beta, bmin)
    [b, db] = b_k(beta, bmin);
    Q = x.^4 + x.^2.*(2.*b.^2 - 2.*a.^2) + a.^4 + 2*a.^2.*b.^2 + b.^4;
    dQx = 4.*x.^3 + 2.*x.*(2.*b.^2 - 2.*a.^2);
    dQa = -4.*x.^2.*a + 4.*a.^3 + 4.*a.*b.^2;
    dQb = db.*(4.*x.^2.*b + 4.*b.^3 + 4.*a.^2.*b);
end