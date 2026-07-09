function [Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma] = psi_fun(x, ak, betak, pk, bmin, sigma)
    n = length(ak);
    m = length(pk);

    Rfun = ones(1, length(x));
    r_k = zeros(n, length(x));
    dRx_k = zeros(n, length(x));
    dRa_k = zeros(n, length(x));
    dRb_k = zeros(n, length(x));

    % Construct the polynomial term
    Palg = poly([pk; -pk; 0]);
    P = polyval(Palg, x);
    dPx = polyval(polyder(Palg), x);

    % Construct the rational term R
    for k=1:n
        [r, rx, ra, rb] = R(x, ak(k), betak(k), bmin);
        Rfun = Rfun.*r;
        r_k(k,:) = r;
        dRx_k(k,:) = rx;
        dRa_k(k,:) = ra;
        dRb_k(k,:) = rb;
    end

    % Construct R's derivative w.r.t. x
    dRx = zeros(1, length(x));
    for k=1:n
        rr = r_k; rr(k,:) = [];
        dRx = dRx + dRx_k(k,:).*prod(rr);
    end

    % Construct the mother wavelet and derivatives
    Psi = P.*Rfun.*exp(-x.^2/sigma^2);

    % Derivatives w.r.t.x
    dPsix = dPx.*Rfun.*exp(-x.^2/sigma^2) + ...
        P.*dRx.*exp(-x.^2/sigma^2) - 2.*x./sigma^2 .* Psi;

    % Derivatives w.r.t. a, b
    dPsia = zeros(n, length(x));
    dPsib = zeros(n, length(x));
    
    for k=1:n
        rr = r_k; rr(k,:) = [];
        dPsia(k,:) = dRa_k(k,:).*prod(rr).*P.*exp(-x.^2/sigma^2);
        dPsib(k,:) = dRb_k(k,:).*prod(rr).*P.*exp(-x.^2/sigma^2);
    end

    % Derivatives w.r.t. p
    dPsip = zeros(m, length(x));
    for k=1:m
        dPp = -( P./( (x - pk(k)).*(x + pk(k)) ) ).*2.*pk(k);
        dPsip(k, :) = dPp.*Rfun.*exp(-x.^2/sigma^2);
    end

    % Derivatives w.r.t. sigma
    dPsiSigma = Psi.*2.*x.^2.*sigma^(-3);
end

